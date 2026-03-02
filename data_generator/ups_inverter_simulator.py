"""
data_generator/ups_inverter_simulator.py

Simulates UPS units and grid-tied inverters feeding the data center.
Includes Grid-Following (GFL) and Grid-Forming (GFM) inverter dynamics,
islanding detection events, and battery state-of-charge tracking.

This bridges directly with your existing gfm-vs-gfl-islanding-sim repo.
"""

import uuid
import random
import math
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from loguru import logger


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GRID_NOMINAL_FREQ_HZ  = 60.0    # Brazil (ANEEL standard)
GRID_NOMINAL_VOLT_V   = 380.0   # Three-phase line voltage
DC_BUS_VOLTAGE_V      = 480.0
BATTERY_CAPACITY_KWH  = 200.0   # Per UPS unit


@dataclass
class UPSTelemetry:
    """Telemetry record for a single UPS unit."""
    event_id:             str
    ups_id:               str
    datacenter_zone:      str
    timestamp_utc:        str

    # Grid interface
    grid_voltage_v:       float
    grid_frequency_hz:    float
    grid_power_factor:    float

    # UPS state
    ups_mode:             str    # "normal" | "bypass" | "battery" | "fault"
    input_power_kw:       float
    output_power_kw:      float
    ups_efficiency:       float  # 0–1

    # Battery
    battery_soc:          float  # 0–1
    battery_voltage_v:    float
    battery_current_a:    float
    battery_temp_c:       float
    estimated_runtime_min:float

    # Alarms
    overload_alarm:       bool
    thermal_alarm:        bool
    battery_alarm:        bool


@dataclass
class InverterTelemetry:
    """Telemetry record for a grid-tied inverter (GFL or GFM)."""
    event_id:              str
    inverter_id:           str
    datacenter_zone:       str
    timestamp_utc:         str

    # Control mode (the key distinction for your thesis)
    control_mode:          str   # "GFL" | "GFM" | "transitioning"
    gfl_pll_locked:        bool
    gfm_virtual_inertia_j: float  # kg·m² equivalent (VSM parameter)

    # Electrical output
    output_active_power_kw:  float
    output_reactive_power_kvar: float
    output_voltage_v:        float
    output_frequency_hz:     float
    thd_percent:             float   # Total Harmonic Distortion

    # Grid event detection
    islanding_detected:    bool
    voltage_deviation_pu:  float   # per-unit deviation from nominal
    freq_deviation_hz:     float
    rocof_hz_per_s:        float   # Rate of Change of Frequency (ROCOF)

    # Health
    dc_link_voltage_v:     float
    junction_temp_c:       float
    efficiency:            float


class UPSSimulator:
    """
    Simulates N UPS units with:
      - Battery SoC dynamics (charge/discharge cycles)
      - Mode transitions (normal → battery → fault)
      - Grid voltage/frequency fluctuations
    """

    def __init__(
        self,
        num_ups: int = 4,
        datacenter_zone: str = "ZONE-A",
        fault_probability: float = 0.005,
    ):
        self.num_ups = num_ups
        self.datacenter_zone = datacenter_zone
        self.fault_probability = fault_probability
        self.units = self._initialize_units()
        logger.info(f"UPSSimulator: {num_ups} units in {datacenter_zone}")

    def _initialize_units(self) -> list[dict]:
        units = []
        for i in range(self.num_ups):
            units.append({
                "ups_id":     f"UPS-{i+1:02d}",
                "soc":        random.uniform(0.85, 1.0),    # Start near full
                "mode":       "normal",
                "capacity_kw": random.choice([100, 150, 200]),
            })
        return units

    def _grid_conditions(self, timestamp: datetime) -> tuple[float, float, float]:
        """Returns (voltage_v, frequency_hz, power_factor) with realistic noise."""
        hour = timestamp.hour + timestamp.minute / 60.0

        # Brazilian grid: slightly lower frequency during peak demand
        freq_offset = -0.15 * math.sin((2 * math.pi / 24) * (hour - 12))
        freq = GRID_NOMINAL_FREQ_HZ + freq_offset + random.gauss(0, 0.02)

        volt_offset = -10 * math.sin((2 * math.pi / 24) * (hour - 14))
        volt = GRID_NOMINAL_VOLT_V + volt_offset + random.gauss(0, 2)

        pf = max(0.85, min(1.0, 0.95 + random.gauss(0, 0.02)))
        return round(volt, 2), round(freq, 4), round(pf, 4)

    def generate_snapshot(self, timestamp: datetime | None = None) -> list[UPSTelemetry]:
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        volt, freq, pf = self._grid_conditions(timestamp)
        records = []

        for unit in self.units:
            # Mode logic
            if random.random() < self.fault_probability:
                unit["mode"] = "battery"
            elif unit["mode"] == "battery" and unit["soc"] < 0.10:
                unit["mode"] = "fault"
            elif unit["mode"] == "battery" and random.random() < 0.05:
                unit["mode"] = "normal"  # Grid restored

            # SoC dynamics
            if unit["mode"] == "battery":
                unit["soc"] = max(0.0, unit["soc"] - random.uniform(0.002, 0.005))
            elif unit["mode"] == "normal":
                unit["soc"] = min(1.0, unit["soc"] + random.uniform(0.001, 0.003))

            soc = unit["soc"]
            cap = unit["capacity_kw"]
            load_frac = random.uniform(0.5, 0.85)
            input_kw  = cap * load_frac if unit["mode"] == "normal" else 0.0
            output_kw = cap * load_frac * random.uniform(0.93, 0.98)
            efficiency = output_kw / max(input_kw, 1e-3)

            batt_v   = 380 + soc * 50
            batt_i   = (output_kw * 1000 / batt_v) if unit["mode"] == "battery" else -abs(random.gauss(10, 2))
            batt_t   = 22 + abs(batt_i) * 0.05 + random.gauss(0, 1)
            runtime  = (BATTERY_CAPACITY_KWH * soc * 60) / max(output_kw, 1)

            records.append(UPSTelemetry(
                event_id=str(uuid.uuid4()),
                ups_id=unit["ups_id"],
                datacenter_zone=self.datacenter_zone,
                timestamp_utc=timestamp.isoformat(),
                grid_voltage_v=volt,
                grid_frequency_hz=freq,
                grid_power_factor=pf,
                ups_mode=unit["mode"],
                input_power_kw=round(input_kw, 2),
                output_power_kw=round(output_kw, 2),
                ups_efficiency=round(min(efficiency, 1.0), 4),
                battery_soc=round(soc, 4),
                battery_voltage_v=round(batt_v, 2),
                battery_current_a=round(batt_i, 2),
                battery_temp_c=round(batt_t, 2),
                estimated_runtime_min=round(runtime, 1),
                overload_alarm=(load_frac > 0.90),
                thermal_alarm=(batt_t > 40),
                battery_alarm=(soc < 0.20),
            ))
        return records


class InverterSimulator:
    """
    Simulates GFL and GFM inverters feeding the data center.

    GFL (Grid-Following):
      - Uses PLL to synchronize to grid voltage
      - Fast response to load changes
      - Loses stability during islanding (PLL can't lock)

    GFM (Grid-Forming):
      - Acts as voltage source (VSM / Droop control)
      - Provides virtual inertia — stabilizes weak grids
      - Can operate autonomously during islanding

    Key metrics for your thesis: ROCOF, frequency nadir, THD, islanding detection
    """

    def __init__(
        self,
        num_inverters: int = 2,
        datacenter_zone: str = "ZONE-A",
        islanding_probability: float = 0.008,
    ):
        self.num_inverters = num_inverters
        self.datacenter_zone = datacenter_zone
        self.islanding_probability = islanding_probability
        self.units = self._initialize_units()
        logger.info(f"InverterSimulator: {num_inverters} inverters in {datacenter_zone}")

    def _initialize_units(self) -> list[dict]:
        units = []
        modes = ["GFL", "GFM"]
        for i in range(self.num_inverters):
            units.append({
                "inverter_id": f"INV-{i+1:02d}",
                "mode":        modes[i % 2],
                "islanding":   False,
                "transition_ticks": 0,
                "rated_kw":    random.choice([250, 500]),
            })
        return units

    def _gfl_dynamics(self, unit: dict, islanding: bool) -> dict:
        """GFL: PLL-based current injection. Unstable under islanding."""
        pll_locked = not islanding
        freq = GRID_NOMINAL_FREQ_HZ + random.gauss(0, 0.03 if pll_locked else 1.5)
        thd  = random.uniform(1.5, 3.5) if pll_locked else random.uniform(8, 20)
        volt = GRID_NOMINAL_VOLT_V + random.gauss(0, 2 if pll_locked else 15)
        rocof = abs(random.gauss(0, 0.05 if pll_locked else 2.5))
        return dict(
            gfl_pll_locked=pll_locked,
            output_frequency_hz=round(freq, 4),
            thd_percent=round(thd, 2),
            output_voltage_v=round(volt, 2),
            rocof_hz_per_s=round(rocof, 4),
        )

    def _gfm_dynamics(self, unit: dict, islanding: bool) -> dict:
        """GFM: Voltage-source with virtual inertia. Stable under islanding."""
        # Virtual inertia constant (VSM parameter H)
        H = random.uniform(3.0, 8.0)
        freq = GRID_NOMINAL_FREQ_HZ + random.gauss(0, 0.05)   # GFM holds frequency
        thd  = random.uniform(2.0, 4.5)                        # Slightly higher THD
        volt = GRID_NOMINAL_VOLT_V + random.gauss(0, 1.5)
        rocof = abs(random.gauss(0, 0.08))                     # Low ROCOF — key GFM advantage
        return dict(
            gfl_pll_locked=True,   # GFM doesn't use PLL
            gfm_virtual_inertia_j=round(H, 3),
            output_frequency_hz=round(freq, 4),
            thd_percent=round(thd, 2),
            output_voltage_v=round(volt, 2),
            rocof_hz_per_s=round(rocof, 4),
        )

    def generate_snapshot(self, timestamp: datetime | None = None) -> list[InverterTelemetry]:
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        records = []

        for unit in self.units:
            # Islanding event
            if random.random() < self.islanding_probability:
                unit["islanding"] = True
                # GFL transitions to GFM after detecting islanding
                if unit["mode"] == "GFL":
                    unit["mode"] = "transitioning"
                    unit["transition_ticks"] = random.randint(3, 8)

            if unit["transition_ticks"] > 0:
                unit["transition_ticks"] -= 1
                if unit["transition_ticks"] == 0:
                    unit["mode"] = "GFM"
                    unit["islanding"] = False

            # Recover from islanding after a few ticks
            if unit["islanding"] and random.random() < 0.15:
                unit["islanding"] = False

            islanding = unit["islanding"]
            mode = unit["mode"]
            rated_kw = unit["rated_kw"]
            load_frac = random.uniform(0.45, 0.85)

            if mode == "GFL":
                dyn = self._gfl_dynamics(unit, islanding)
                virtual_inertia = 0.0
            elif mode == "GFM":
                dyn = self._gfm_dynamics(unit, islanding)
                virtual_inertia = dyn.pop("gfm_virtual_inertia_j", 5.0)
            else:  # transitioning
                dyn = self._gfl_dynamics(unit, True)
                virtual_inertia = random.uniform(0.5, 3.0)

            active_kw = rated_kw * load_frac * (0.6 if islanding and mode == "GFL" else 1.0)
            reactive_kvar = active_kw * random.uniform(0.1, 0.35)
            volt_pu = dyn["output_voltage_v"] / GRID_NOMINAL_VOLT_V
            freq_dev = dyn["output_frequency_hz"] - GRID_NOMINAL_FREQ_HZ

            records.append(InverterTelemetry(
                event_id=str(uuid.uuid4()),
                inverter_id=unit["inverter_id"],
                datacenter_zone=self.datacenter_zone,
                timestamp_utc=timestamp.isoformat(),
                control_mode=mode,
                gfl_pll_locked=dyn["gfl_pll_locked"],
                gfm_virtual_inertia_j=virtual_inertia,
                output_active_power_kw=round(active_kw, 2),
                output_reactive_power_kvar=round(reactive_kvar, 2),
                output_voltage_v=dyn["output_voltage_v"],
                output_frequency_hz=dyn["output_frequency_hz"],
                thd_percent=dyn["thd_percent"],
                islanding_detected=islanding,
                voltage_deviation_pu=round(abs(volt_pu - 1.0), 5),
                freq_deviation_hz=round(freq_dev, 5),
                rocof_hz_per_s=dyn["rocof_hz_per_s"],
                dc_link_voltage_v=round(DC_BUS_VOLTAGE_V + random.gauss(0, 5), 2),
                junction_temp_c=round(40 + load_frac * 30 + random.gauss(0, 2), 2),
                efficiency=round(random.uniform(0.95, 0.99), 4),
            ))
        return records


# ---------------------------------------------------------------------------
# Quick standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    from dataclasses import asdict

    ups_sim = UPSSimulator(num_ups=4)
    inv_sim = InverterSimulator(num_inverters=2, islanding_probability=0.5)

    print("=== UPS Snapshot ===")
    for rec in ups_sim.generate_snapshot():
        print(json.dumps(asdict(rec), indent=2))

    print("\n=== Inverter Snapshot ===")
    for rec in inv_sim.generate_snapshot():
        print(json.dumps(asdict(rec), indent=2))
