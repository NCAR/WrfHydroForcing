#!/usr/bin/env python
import os
import sys
import enum
from strenum import StrEnum
from enum import IntEnum

class ForcingEnum(IntEnum):
    NLDAS = 1 #GRIB retrospective files
    NARR = 2 #GRIB retrospective files
    GFS_GLOBAL = 3 #GRIB2 Global production files on the full gaussian grid
    NAM_NEST_CONUS = 4 #Nest GRIB2 Conus production files
    HRRR = 5 #GRIB2 Conus production files
    RAP = 6 #GRIB2 Conus 13km production files
    CFS_V2 = 7 #6-hourly GRIB2 Global production files
    WRF_NEST_HI = 8 #GRIB2 Hawaii nest files
    GFS_GLOBAL_25 = 9 #GRIB2 Global production files on 0.25 degree lat/lon grids. 
    CUSTOM_1 = 10 #Custom NetCDF hourly forcing files
    CUSTOM_2 = 11 #NetCDF hourly forcing files
    CUSTOM_3 = 12 #NetCDF hourly forcing files
    NAM_NEST_HI = 13 #3-km NAM Nest.
    NAM_NEST_PR = 14 #3-km NAM Nest.
    NAM_NEST_AK = 15 #3-km Alaska Nest
    NAM_NEST_HI_RAD = 16 #NAM_Nest_3km_Hawaii_Radiation-Only
    NAM_NEST_PR_RAD = 17 #NAM_Nest_3km_PuertoRico_Radiation-Only
    WRF_ARW_PR = 18 #GRIB2 PuertoRico
    HRRR_AK = 19 #HRRR GRIB2 Alaska production files 
    HRRR_AK_EXT = 20 #ExtAna HRRR AK FE output

class ForcingTypeEnum(StrEnum):
    GRIB1  = "GRIB1"
    GRIB2  = "GRIB2"
    NETCDF = "NETCDF"

class RegriddingOptEnum(StrEnum):
    ESMF_BILINEAR              = "ESMF_BILINEAR"
    ESMF_NEAREST_NEIGHBOR      = "ESMF_NEAREST_NEIGHBOR"
    ESMF_CONSERVATIVE_BILINEAR = "ESMF_CONSERVATIVE_BILINEAR"

class TemporalInterpEnum(StrEnum):
    NONE              = "NONE"
    NEAREST_NEIGHBOR  = "NEAREST_NEIGHBOR"
    LINEAR_WEIGHT_AVG = "LINEAR_WEIGHT_AVG"

class BiasCorrTempEnum(StrEnum):
    NONE   = "NONE"
    CFS_V2 = "CFS_V2"
    CUSTOM = "CUSTOM"
    GFS    = "GFS"
    HRRR   = "HRRR"

class BiasCorrPressEnum(StrEnum):
    NONE   = "NONE"
    CFS_V2 = "CFS_V2"

class BiasCorrHumidEnum(StrEnum):
    NONE   = "NONE"
    CFS_V2 = "CFS_V2"
    CUSTOM = "CUSTOM"

class BiasCorrWindEnum(StrEnum):
    NONE   = "NONE"
    CFS_V2 = "CFS_V2"
    CUSTOM = "CUSTOM"
    GFS    = "GFS"
    HRRR   = "HRRR"

class BiasCorrSwEnum(StrEnum):
    NONE   = "NONE"
    CFS_V2 = "CFS_V2"
    CUSTOM = "CUSTOM"

class BiasCorrLwEnum(StrEnum):
    NONE   = "NONE"
    CFS_V2 = "CFS_V2"
    CUSTOM = "CUSTOM"
    GFS    = "GFS"

class BiasCorrPrecipEnum(StrEnum):
    NONE   = "NONE"
    CFS_V2 = "CFS_V2"

class DownScaleTempEnum(StrEnum):
    NONE           = "NONE"
    LAPSE_675      = "LAPSE_675"
    LAPSE_PRE_CALC = "LAPSE_PRE_CALC"

class DownScalePressEnum(StrEnum):
    NONE = "NONE"
    ELEV = "ELEV"

class DownScaleSwEnum(StrEnum):
    NONE = "NONE"
    ELEV = "ELEV"

class DownScalePrecipEnum(StrEnum):
    NONE   = "NONE"
    NWM_MM = "NWM_MM"

class DownScaleHumidEnum(StrEnum):
    NONE              = "NONE"
    REGRID_TEMP_PRESS = "REGRID_TEMP_PRESS"

class OutputFloatEnum(StrEnum):
    SCALE_OFFSET = "SCALE_OFFSET"
    FLOAT        = "FLOAT"

class SuppForcingRqiMethodEnum(StrEnum):
    NONE = "NONE"
    MRMS = "MRMS"
    NWM  = "NWM"

class SuppForcingPcpEnum(StrEnum):
    MRMS          = "MRMS"
    MRMS_GAGE     = "MRMS_GAGE"
    WRF_ARW_HI    = "WRF_ARW_HI"
    WRF_ARW_PR    = "WRF_ARW_PR"
    MRMS_CONUS_MS = "MRMS_CONUS_MS"
    MRMS_HI_MS    = "MRMS_HI_MS"
    MRMS_SBCV2    = "MRMS_SBCV2"
    AK_OPT1       = "AK_OPT1"
    AK_OPT2       = "AK_OPT2"
    AK_MRMS       = "AK_MRMS"
    AK_NWS_IV     = "AK_NWS_IV"

class RegriddingOptEnum(IntEnum):
    ESMF_BILINEAR = 1
    ESMF_NEAREST_NEIGHBOR = 2
    ESMF_CONSERVATIVE_BILINEAR = 3

class TemporalInterpEnum(IntEnum):
    NONE = 0
    NEAREST_NEIGHBOR = 1
    LINEAR_WEIGHT_AVG = 2

class BiasCorrTempEnum(IntEnum):
    NONE = 0
    CFS_V2 = 1
    CUSTOM = 2
    GFS = 3
    HRRR = 4

class BiasCorrPressEnum(IntEnum):
    NONE = 0
    CFS_V2 = 1

class BiasCorrHumidEnum(IntEnum):
    NONE = 0
    CFS_V2 = 1
    CUSTOM = 2

class BiasCorrWindEnum(IntEnum):
    NONE = 0
    CFS_V2 = 1
    CUSTOM = 2
    GFS = 3
    HRRR = 4

class BiasCorrSwEnum(IntEnum):
    NONE = 0
    CFS_V2 = 1
    CUSTOM = 2

class BiasCorrLwEnum(IntEnum):
    NONE = 0
    CFS_V2 = 1
    CUSTOM = 2
    GFS = 3

class BiasCorrPrecipEnum(IntEnum):
    NONE = 0
    CFS_V2 = 1

class DownScaleTempEnum(IntEnum):
    NONE = 0
    LAPSE_675 = 1
    LAPSE_PRE_CALC = 2

class DownScalePressEnum(IntEnum):
    NONE = 0
    ELEV = 1

class DownScaleSwEnum(IntEnum):
    NONE = 0
    ELEV = 1

class DownScalePrecipEnum(IntEnum):
    NONE = 0
    NWM_MM = 1

class DownScaleHumidEnum(IntEnum):
    NONE = 0
    REGRID_TEMP_PRESS = 1

class OutputFloatEnum(IntEnum):
    SCALE_OFFSET = 0
    FLOAT = 1

class SuppForcingPcpEnum(IntEnum):
    MRMS = 1
    MRMS_GAGE = 2
    WRF_ARW_HI = 3
    WRF_ARW_PR = 4
    MRMS_CONUS_MS = 5
    MRMS_HI_MS = 6
    MRMS_SBCV2 = 7
    AK_OPT1 = 8
    AK_OPT2 = 9
    AK_MRMS = 10
    AK_NWS_IV = 11

class SuppForcingRqiMethodEnum(IntEnum):
    NONE = 0
    MRMS = 1
    NWM = 2
