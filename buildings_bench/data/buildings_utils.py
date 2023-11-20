# utility functions for building characteristics loading
# Author: Zhaonan Li zli4@nrel.gov

res_chars = [
    "in.bedrooms",
    "in.cec_climate_zone",
    "in.ceiling_fan",
    "in.census_division",
    "in.census_division_recs",
    "in.census_region",
    "in.clothes_dryer",
    "in.clothes_washer",
    "in.clothes_washer_presence",
    "in.cooking_range",
    "in.cooling_setpoint",
    "in.cooling_setpoint_offset_magnitude",
    "in.dishwasher",
    "in.ducts",
    "in.geometry_floor_area",
    "in.geometry_floor_area_bin",
    "in.geometry_foundation_type",
    "in.geometry_garage",
    "in.geometry_stories",
    "in.geometry_stories_low_rise",
    "in.geometry_wall_exterior_finish",
    "in.geometry_wall_type",
    "in.geometry_wall_type_and_exterior_finish",
    "in.has_pv",
    "in.heating_fuel",
    "in.heating_setpoint",
    "in.heating_setpoint_has_offset",
    "in.heating_setpoint_offset_magnitude",
    "in.hvac_cooling_efficiency",
    "in.hvac_heating_efficiency",
    "in.hvac_shared_efficiencies",
    "in.infiltration",
    "in.insulation_slab",
    "in.insulation_wall",
    "in.lighting",
    "in.misc_extra_refrigerator",
    "in.misc_freezer",
    "in.misc_gas_fireplace",
    "in.misc_gas_grill",
    "in.misc_gas_lighting",
    "in.misc_hot_tub_spa",
    "in.misc_pool_pump",
    "in.misc_pool_heater",
    "in.misc_well_pump",
    "in.natural_ventilation",
    "in.neighbors",
    "in.occupants",
    "in.plug_loads",
    "in.refrigerator",
    "in.water_heater_efficiency",
    "in.windows"
]

# categorical characteristics
com_chars = [
    "in.building_subtype",
    "in.building_type",
    "in.rotation",
    "in.number_of_stories",
    "in.sqft",
    "in.hvac_system_type",
    "in.weekday_operating_hours",
    "in.weekday_opening_time",
    "in.weekend_operating_hours",
    "in.weekend_opening_time",
    "in.heating_fuel",
    "in.service_water_heating_fuel",
    "stat.average_boiler_efficiency",
    "stat.average_gas_coil_efficiency"
]

total_chars = res_chars + com_chars