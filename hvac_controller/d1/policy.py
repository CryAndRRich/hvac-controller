"""Direction 1, step 6: the policy, one code path for training (batched, GPU) and for a submission (one row, CPU).

``Layout`` reads one building's index maps from the metadata the organisers' gateway hands a policy (nothing else:
no simulator object, no building or weather identity). ``Controller`` holds the weights, shared by every building:

- features: per VAV its zone temperature and comfort band (the setpoint sensors), the PI integral; globally the
  time features, comfort flags, occupant count, outside temperature and hot-water tank temperature. They are decoded
  exactly as the environment encoded them: a field is normalised only when its measurement name (the field name
  without the device id ``VAV_<n>``, ``ahs`` or ``hws``) is in the normaliser map, so the zone temperature is
  normalised but the setpoint sensors, the AHUs' outside-air field (``ahu_<n>_outside_…``), the tank and the
  auxiliary features are raw;
- encoders: VAV -> AHU (mean of its VAVs) -> global (mean of the AHUs), Linear + LayerNorm + SiLU;
- supervisory heads (final layers zero at initialisation, so the priors below are the initial policy):
  per VAV a target temperature inside its band (initially the band centre) and gains on the damper and reheat;
  per AHU a run command, supply temperature, static pressure and recirculation; for the plant the water setpoint,
  pump differential pressure and pump run command. Priors (as K10): AHUs and pump on while the comfort mode is on
  or due within the hour, supply 288 K, static pressure ~285 Pa, recirculation ~0.67. The run commands stay on in
  those hours; the network can only add running time outside them (pre-heating, pre-cooling). The pilot
  (notebook 21, 2026-10-20) let it switch units off in occupied hours too: after ~50 updates the straight-through
  gradient (always "a little less air saves energy") flipped them off, comfort collapsed and the saturated sigmoid
  never switched them back;
- a differentiable PI loop per VAV towards its target (K10's law with tanh in place of the clip): damper
  min + (1 - min) |u|, reheat max(0, -u), each times its learnt gain; Kp and Ki are learnt (initially 0.1, 0.005);
  the integral runs while the comfort mode is on or due, and is zero otherwise;
- every continuous command is a sigmoid into its feasible native range (static pressure from the 200 Pa floor,
  recirculation up to its cap, damper from its floor), so no device clamp or floor ever binds;
- run commands: 0 / 1 in the forward pass, the sigmoid's gradient in the backward pass (straight-through). The
  replica's relaxed plant fed 0 / 1 computes exactly what the exact plant does, so training and scoring run the same
  forward pass.

- memory (``memory=True``): exponential averages of the outside temperature (1 h, 6 h, 24 h) and of each zone's
  temperature (1 h, 6 h), carried with the PI integrals; the first step of an episode sets them to the observed
  values. They let the policy see the weather's trend and each zone's drift, which one observation does not hold.

Layouts are per row: one building's (``Layout.tensors``, leading dimension 1) or a padded batch of several
buildings' (``stack_layouts``: VAVs, AHUs and action columns padded to the largest, with masks), so one batch can
mix buildings of different sizes.

``KagglePolicy`` is the gateway's ``Policy(building_meta_data, action_spec)`` around it.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import torch
from torch import nn

GLOBAL_FIELDS = ("hod_sin_000", "hod_cos_000", "dow_sin_000", "dow_cos_000", "comfort_mode_now", "comfort_mode_soon",
                 "num_occupants")
N_GLOBAL = len(GLOBAL_FIELDS) + 2  # + outside and tank temperature
N_VAV = 6
TAU_OUT, TAU_ZONE = (3600.0, 21600.0, 86400.0), (3600.0, 21600.0)  # memory time constants, seconds
DT = 300.0
# Continuous commands in the order the controller produces them: per AHU (A each), plant, per VAV (V each).
CONT_GROUPS = ("ahu_sp", "ahu_p", "ahu_r", "w_sp", "w_dp", "damper", "valve")


def _numeric(name: str) -> int:
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else 0


def field_constants(name: str, constants: dict) -> tuple[float, float]:
    """(mean, std) the environment normalised an observation field with; (0, 1) for a raw one."""
    m = re.match(r"^(VAV_\d+|ahs|hws)_(.+)$", name)
    c = constants.get(m.group(2) if m else name)
    if c is None:
        return 0.0, 1.0
    if c["sample_variance"] <= 0.0:
        raise ValueError(f"{name}: zero normaliser variance, the environment reports 0 and nothing is decodable")
    return float(c["sample_mean"]), float(math.sqrt(c["sample_variance"]))


class Layout:
    """One building's observation and action index maps, from the gateway's metadata."""

    def __init__(self, meta: dict):
        names = list(meta["observation_field_names"])
        consts = meta["observation_normalization_metadata"]
        zai = meta["zone_action_indices"]
        self.vav_ids = sorted(zai, key=_numeric)
        self.ahu_ids = sorted(meta["ahu_zone_assignments"], key=_numeric)
        V, A = len(self.vav_ids), len(self.ahu_ids)
        self.V, self.A = V, A

        def fields(suffix):
            idx = [names.index(f"{v}_{suffix}") for v in self.vav_ids]
            return idx, np.array([field_constants(names[i], consts) for i in idx])

        self.i_temp, self.c_temp = fields("zone_air_temperature_sensor")
        self.i_heat, self.c_heat = fields("zone_air_heating_temperature_setpoint_sensor")
        self.i_cool, self.c_cool = fields("zone_air_cooling_temperature_setpoint_sensor")
        self.i_glob = [names.index(f) for f in GLOBAL_FIELDS]
        oat = f"ahs_{self.ahu_ids[0]}_outside_air_temperature_sensor"
        tank = next(n for n in names if n.endswith("supply_water_temperature_sensor"))
        self.i_temps = [names.index(oat), names.index(tank)]
        self.c_temps = np.array([field_constants(oat, consts), field_constants(tank, consts)])
        vav_of_room = {r.replace("room_", "VAV_"): a for a, ahu in enumerate(self.ahu_ids)
                       for r in meta["ahu_zone_assignments"][ahu]}
        self.vav_ahu = np.array([vav_of_room[v] for v in self.vav_ids])
        # Actions: flat column of each command in CONT_GROUPS order, and its native and normalised ranges.
        an = meta["action_normalization_metadata"]
        p_idx = list(meta["pressure_indices"])
        cols = {"ahu_sp": list(meta["temp_indices"]), "ahu_p": p_idx[:A], "ahu_r": list(meta["recirculation_indices"]),
                "w_sp": list(meta["water_indices"]), "w_dp": p_idx[A:],
                "damper": [zai[v]["supply_air_damper_percentage_command"] for v in self.vav_ids],
                "valve": [zai[v]["reheat_valve_percentage_command"] for v in self.vav_ids]}
        keys = {"ahu_sp": [f"{a}_supply_air_temperature_setpoint" for a in self.ahu_ids],
                "ahu_p": [f"{a}_supply_air_static_pressure_setpoint" for a in self.ahu_ids],
                "ahu_r": [f"{a}_recirculation_percentage_command" for a in self.ahu_ids],
                "w_sp": ["supply_water_temperature_setpoint"], "w_dp": ["differential_pressure_setpoint"],
                "damper": ["supply_air_damper_percentage_command"] * V,
                "valve": ["reheat_valve_percentage_command"] * V}
        self.cont_cols, self.cont_targets, ranges = [], [], []
        for g in CONT_GROUPS:
            if len(cols[g]) != len(keys[g]):
                raise ValueError(f"{g}: {len(cols[g])} action columns for {len(keys[g])} devices")
            for j, (c, k) in enumerate(zip(cols[g], keys[g])):
                r = an[k]
                self.cont_cols.append(c)
                self.cont_targets.append((g, j))
                ranges.append((r["min_native_value"], r["max_native_value"], r["min_normalized_value"],
                               r["max_normalized_value"]))
        self.cont_ranges = np.array(ranges, dtype=np.float64)  # (n_cont, 4): native lo, hi, normalised lo, hi
        self.n_cont, self.n_disc = int(meta["n_continuous_action"]), int(meta["n_discrete_action"])
        if sorted(self.cont_cols) != list(range(self.n_cont)) or self.n_disc != A + 1:
            raise ValueError("the action columns do not cover the action exactly once")
        self.disc_cols = list(range(A)) + [self.n_disc - 1]  # run command of AHU i, then the pump

    def check(self, spec) -> None:
        """Against the replica's spec (read from the simulator's own objects): every field this layout decodes has the
        constants the environment uses, and every action column drives the device and range the layout thinks."""
        if (self.V, self.A) != (spec.n_vavs, spec.n_ahus):
            raise AssertionError(f"layout V={self.V} A={self.A}, environment V={spec.n_vavs} A={spec.n_ahus}")
        decoded = [(self.i_temp, self.c_temp), (self.i_heat, self.c_heat), (self.i_cool, self.c_cool),
                   (self.i_temps, self.c_temps), (self.i_glob, [(0.0, 1.0)] * len(self.i_glob))]
        for idx, consts in decoded:
            for i, c in zip(idx, consts):
                want = spec.obs[i][3:5]  # the environment's (mean, std); (0, 1) for a raw field
                if not np.allclose(c, want):
                    raise AssertionError(f"field {spec.field_names[i]}: layout decodes with {tuple(c)}, environment {want}")
        vav_index = {v: j for j, v in enumerate(spec.vav_ids)}
        for k, (c, (g, j)) in enumerate(zip(self.cont_cols, self.cont_targets)):
            target, dev = spec.cont[c][:2]
            want = (vav_index[self.vav_ids[j]] if g in ("damper", "valve")
                    else _numeric(self.ahu_ids[j]) - 1 if g.startswith("ahu") else 0)
            if (target, dev) != (g, want):
                raise AssertionError(f"action column {c}: layout {g}[{want}], environment {target}[{dev}]")
            if not np.allclose(spec.cont[c][2:6], self.cont_ranges[k][[2, 3, 0, 1]]):
                raise AssertionError(f"action column {c}: ranges {self.cont_ranges[k]} vs {spec.cont[c][2:6]}")
        for a, c in enumerate(self.disc_cols):
            want = ("ahu_on", _numeric(self.ahu_ids[a]) - 1) if a < self.A else ("w_on", 0)
            if tuple(spec.disc[c][:2]) != want:
                raise AssertionError(f"discrete column {c}: layout {want}, environment {tuple(spec.disc[c][:2])}")

    def padded(self, V: int, A: int, n_cont: int, n_disc: int) -> dict:
        """NumPy arrays of this building padded to V VAVs, A AHUs and the given action widths.

        The controller's native command vector is [supply temperature (A) | pressure (A) | recirculation (A) | water
        setpoint, pump pressure (2) | damper (V) | valve (V)]; inv_cont maps each action column to its entry there,
        inv_disc each discrete column to [AHU run (A) | pump run (1)]; padded columns are masked to 0."""
        v, a = self.V, self.A
        off = {"ahu_sp": 0, "ahu_p": A, "ahu_r": 2 * A, "w_sp": 3 * A, "w_dp": 3 * A + 1, "damper": 3 * A + 2,
               "valve": 3 * A + 2 + V}
        n_nat = 3 * A + 2 + 2 * V
        lo, hi, nlo, nhi = np.zeros(n_nat), np.ones(n_nat), -np.ones(n_nat), np.ones(n_nat)
        inv, c_mask = np.zeros(n_cont, dtype=np.int64), np.zeros(n_cont)
        for k, (c, (g, j)) in enumerate(zip(self.cont_cols, self.cont_targets)):
            pos = off[g] + j
            lo[pos], hi[pos], nlo[pos], nhi[pos] = self.cont_ranges[k]
            inv[c], c_mask[c] = pos, 1.0
        inv_d, d_mask = np.zeros(n_disc, dtype=np.int64), np.zeros(n_disc)
        for i, c in enumerate(self.disc_cols):
            inv_d[c], d_mask[c] = (i if i < a else A), 1.0
        pool, onehot = np.zeros((A, V)), np.zeros((V, A))
        pool[self.vav_ahu, np.arange(v)] = 1.0
        pool /= np.maximum(pool.sum(1, keepdims=True), 1.0)
        onehot[np.arange(v), self.vav_ahu] = 1.0

        def pad(x, n, fill):
            x = np.asarray(x)
            return np.concatenate([x, np.repeat(np.asarray(fill, dtype=x.dtype)[None], n - len(x), 0)]) if n > len(x) else x

        unit = np.array([0.0, 1.0])
        return {"i_temp": pad(self.i_temp, V, 0), "i_heat": pad(self.i_heat, V, 0), "i_cool": pad(self.i_cool, V, 0),
                "c_temp": pad(self.c_temp, V, unit), "c_heat": pad(self.c_heat, V, unit),
                "c_cool": pad(self.c_cool, V, unit), "i_glob": np.asarray(self.i_glob), "i_temps": np.asarray(self.i_temps),
                "c_temps": self.c_temps, "pool": pool, "onehot": onehot, "v_mask": pad(np.ones(v), V, 0.0),
                "a_mask": pad(np.ones(a), A, 0.0), "inv_cont": inv, "c_mask": c_mask, "inv_disc": inv_d,
                "d_mask": d_mask, "native_lo": lo, "native_hi": hi, "norm_lo": nlo, "norm_hi": nhi}

    def tensors(self, device) -> dict:
        """This building's layout for a batch of its rows (leading dimension 1)."""
        return _to_tensors([self.padded(self.V, self.A, self.n_cont, self.n_disc)], np.zeros(1, dtype=np.int64),
                           self.V, self.A, device)


def stack_layouts(layouts: list[Layout], row_s, device) -> dict:
    """A padded per-row layout for a batch mixing buildings: row i has layouts[row_s[i]]."""
    V, A = max(l.V for l in layouts), max(l.A for l in layouts)
    n_cont, n_disc = max(l.n_cont for l in layouts), max(l.n_disc for l in layouts)
    return _to_tensors([l.padded(V, A, n_cont, n_disc) for l in layouts], np.asarray(row_s), V, A, device)


def _to_tensors(pads: list[dict], row_s, V: int, A: int, device) -> dict:
    out = {"V": V, "A": A}
    for k in pads[0]:
        v = np.stack([p[k] for p in pads])[row_s]
        dt = torch.long if v.dtype.kind in "iu" else torch.float32
        out[k] = torch.as_tensor(v, dtype=dt, device=device)
    return out


def _take(x, idx):
    """x (B, n) at columns idx (1 or B, k)."""
    return torch.gather(x, 1, idx.expand(x.shape[0], -1))


def _mlp(n_in: int, width: int, n_out: int | None = None) -> nn.Sequential:
    layers = [nn.Linear(n_in, width), nn.LayerNorm(width), nn.SiLU(), nn.Linear(width, width), nn.LayerNorm(width),
              nn.SiLU()]
    if n_out is not None:
        head = nn.Linear(width, n_out)
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        layers.append(head)
    return nn.Sequential(*layers)


def _hard(p):
    """0 / 1 forward, the gradient of p backward (p - p.detach() is exactly zero)."""
    return (p > 0.5).to(p.dtype) + (p - p.detach())


class Controller(nn.Module):
    def __init__(self, width: int = 64, p_floor: float = 200.0, max_recirc: float = 0.7, min_damper: float = 0.1,
                 kp: float = 0.1, ki: float = 0.005, memory: bool = True):
        super().__init__()
        self.config = {"width": width, "p_floor": p_floor, "max_recirc": max_recirc, "min_damper": min_damper,
                       "kp": kp, "ki": ki, "memory": memory}
        self.p_floor, self.max_recirc, self.min_damper, self.memory = p_floor, max_recirc, min_damper, memory
        self.i_max = 1.0 / ki
        self.n_vav = N_VAV + (len(TAU_ZONE) if memory else 0)
        self.n_glob = N_GLOBAL + (len(TAU_OUT) if memory else 0)
        n_v = self.n_vav + self.n_glob
        self.enc_v = _mlp(n_v, width)
        self.enc_a = _mlp(width + self.n_glob, width)
        self.enc_g = _mlp(width + self.n_glob, width)
        self.dec_v = _mlp(3 * width, width, 3)
        self.dec_a = _mlp(2 * width, width, 4)
        self.dec_p = _mlp(width, width, 3)
        self.log_kp = nn.Parameter(torch.tensor(math.log(kp)))
        self.log_ki = nn.Parameter(torch.tensor(math.log(ki)))
        # Head offsets, the priors: AHU (run: set by the comfort flags in act, supply 288 K of 285-305, pressure
        # ~285 Pa, recirculation ~0.67); plant (water 330 K, pump dp near its maximum, pump run as the AHUs).
        self.register_buffer("ahu_bias", torch.tensor([0.0, math.log(3.0 / 17.0), -3.0, 3.0]))
        self.register_buffer("plant_bias", torch.tensor([0.0, 3.0, 0.0]))
        self.register_buffer("alpha_out", torch.tensor([1.0 - math.exp(-DT / t) for t in TAU_OUT]))
        self.register_buffer("alpha_zone", torch.tensor([1.0 - math.exp(-DT / t) for t in TAU_ZONE]))

    @staticmethod
    def init_mem(rows: int, lay: dict, device=None) -> tuple:
        """Episode-start memory: PI integrals (B, V), zone averages (B, V, 2), outside averages and step count (B, 4)."""
        V = lay["V"]
        return (torch.zeros(rows, V, device=device), torch.zeros(rows, V, len(TAU_ZONE), device=device),
                torch.zeros(rows, len(TAU_OUT) + 1, device=device))

    @staticmethod
    def reset_mem(mem: tuple, done) -> tuple:
        """Rows whose episode ended back to the episode-start memory."""
        return tuple(torch.where(done.view(-1, *([1] * (m.dim() - 1))), torch.zeros_like(m), m) for m in mem)

    def _decode(self, obs, lay: dict):
        def dec(idx, c):
            return _take(obs, idx) * c[..., 1] + c[..., 0]

        T, lo, hi = dec(lay["i_temp"], lay["c_temp"]), dec(lay["i_heat"], lay["c_heat"]), dec(lay["i_cool"], lay["c_cool"])
        return T, lo, hi, _take(obs, lay["i_glob"]), dec(lay["i_temps"], lay["c_temps"])

    def _remember(self, mem: tuple, T, t_out) -> tuple:
        """Memory after seeing this step's zone and outside temperatures (the first step of an episode sets it)."""
        I, m_v, m_g = mem
        n = m_g[:, -1:]
        a_out = torch.maximum(self.alpha_out, 1.0 / (n + 1.0))
        a_zone = torch.maximum(self.alpha_zone, 1.0 / (n[:, :, None] + 1.0))
        m_out = m_g[:, :-1] + a_out * (t_out - m_g[:, :-1])
        m_v = m_v + a_zone * (T[..., None] - m_v)
        return I, m_v, torch.cat([m_out, n + 1.0], 1)

    def features(self, obs, mem: tuple, lay: dict):
        """Decoded per-VAV (B, V, n_vav) and global (B, n_glob) features, kelvin temperatures, the band and the
        memory updated with this observation."""
        T, lo, hi, g, temps = self._decode(obs, lay)
        mem = self._remember(mem, T, temps[:, :1])
        I, m_v, m_g = mem
        glob = [g[:, :6], torch.log1p(g[:, 6:7].clamp(min=0.0)) / 5.0, (temps[:, :1] - 295.0) / 10.0,
                (temps[:, 1:] - 330.0) / 20.0]
        vav = [(T - 295.0) / 5.0, (lo - 295.0) / 5.0, (hi - 295.0) / 5.0, (T - lo) / 2.0, (T - hi) / 2.0,
               I * self.config["ki"]]
        if self.memory:
            glob.append((m_g[:, :-1] - 295.0) / 10.0)
            vav += [(m_v[..., j] - T) / 2.0 for j in range(m_v.shape[2])]
        return torch.stack(vav, 2), torch.cat(glob, 1), T, lo, hi, mem

    def act(self, obs, mem: tuple, lay: dict):
        """obs (B, n_obs), memory -> continuous (B, n_cont) and discrete (B, n_disc) actions, memory'."""
        V, A = lay["V"], lay["A"]
        vav, glob, T, lo, hi, (I, m_v, m_g) = self.features(obs, mem, lay)
        B = obs.shape[0]
        gv = glob[:, None, :].expand(B, V, glob.shape[1])
        h_v = self.enc_v(torch.cat([vav, gv], 2))
        h_a = self.enc_a(torch.cat([torch.matmul(lay["pool"], h_v), glob[:, None, :].expand(B, A, glob.shape[1])], 2))
        a_mask = lay["a_mask"][..., None]
        h_g = self.enc_g(torch.cat([(h_a * a_mask).sum(1) / a_mask.sum(1), glob], 1))
        z_v = self.dec_v(torch.cat([h_v, torch.matmul(lay["onehot"], h_a), h_g[:, None, :].expand(B, V, h_g.shape[1])], 2))
        z_a = self.dec_a(torch.cat([h_a, h_g[:, None, :].expand(B, A, h_g.shape[1])], 2)) + self.ahu_bias
        z_p = self.dec_p(h_g) + self.plant_bias
        active = torch.maximum(glob[:, 4:5], glob[:, 5:6])  # comfort mode now or within the hour
        # Per-VAV PI towards a target inside the band.
        target = lo + (hi - lo) * torch.sigmoid(z_v[..., 0])
        err = T - target
        I = active * torch.clamp(I + err, -self.i_max, self.i_max)
        u = torch.tanh(torch.exp(self.log_kp) * err + torch.exp(self.log_ki) * I)
        damper = self.min_damper + (1.0 - self.min_damper) * torch.clamp(u.abs() * 2.0 * torch.sigmoid(z_v[..., 1]),
                                                                          max=1.0)
        valve = torch.clamp(torch.relu(-u) * 2.0 * torch.sigmoid(z_v[..., 2]), max=1.0)
        # AHUs and plant: sigmoids into the feasible native ranges.
        lo_n, hi_n = lay["native_lo"], lay["native_hi"]
        s_a, s_p = torch.sigmoid(z_a), torch.sigmoid(z_p)
        idle = 1.0 - active  # the network's run logits act only outside the comfort hours
        run_a = _hard(torch.sigmoid(6.0 * active - 3.0 + idle * z_a[..., 0]))
        pump = _hard(torch.sigmoid(6.0 * active - 3.0 + idle * z_p[:, 2:3]))
        sp = lo_n[:, :A] + (hi_n[:, :A] - lo_n[:, :A]) * s_a[..., 1]
        p = self.p_floor + (hi_n[:, A:2 * A] - self.p_floor) * s_a[..., 2]
        r = lo_n[:, 2 * A:3 * A] + (self.max_recirc - lo_n[:, 2 * A:3 * A]) * s_a[..., 3]
        w = lo_n[:, 3 * A:3 * A + 2] + (hi_n[:, 3 * A:3 * A + 2] - lo_n[:, 3 * A:3 * A + 2]) * s_p[:, :2]
        native = torch.cat([sp, p, r, w, damper, valve], 1)
        a = (native - lo_n) / (hi_n - lo_n) * (lay["norm_hi"] - lay["norm_lo"]) + lay["norm_lo"]
        a_c = _take(a, lay["inv_cont"]) * lay["c_mask"]
        a_d = _take(torch.cat([run_a, pump], 1), lay["inv_disc"]) * lay["d_mask"]
        return a_c, a_d, (I, m_v, m_g)


def find_weights(root: Path = Path("/kaggle/input"), name: str = "d1_policy.pt") -> Path:
    return next(iter(sorted(root.rglob(name))))


def save_policy(ctrl: Controller, path: Path) -> None:
    torch.save({"config": ctrl.config, "state_dict": ctrl.state_dict()}, path)


def load_policy(path: Path, device="cpu") -> Controller:
    ck = torch.load(path, map_location=device)
    ctrl = Controller(**ck["config"]).to(device)
    ctrl.load_state_dict(ck["state_dict"])
    return ctrl.eval()


class KagglePolicy:
    """The gateway's policy: one building, one observation per call, CPU."""

    def __init__(self, building_meta_data, action_spec, weights: Path | None = None):
        self.ctrl = load_policy(weights or find_weights())
        self.lay = Layout(building_meta_data).tensors("cpu")
        self.mem = Controller.init_mem(1, self.lay)

    def get_action(self, obs):
        with torch.no_grad():
            a_c, a_d, self.mem = self.ctrl.act(torch.as_tensor(np.asarray(obs), dtype=torch.float32)[None], self.mem,
                                               self.lay)
        return {"continuous_action": np.clip(a_c[0].numpy(), -1.0, 1.0).astype(np.float32),
                "discrete_action": a_d[0].numpy().round().astype(np.int32)}
