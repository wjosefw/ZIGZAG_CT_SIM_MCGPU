# >>>> INPUT FILE FOR MC-GPU v1.5 VICTRE-DBT >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#
# MC-GPU simulation parameters for the Bern parcel CT scan.
# This file is the single source of truth for all simulation geometry and settings.
# It is never modified at runtime — sweep generation reads from here and writes .in files.


# [SECTION SIMULATION CONFIG v.2009-05-12]

N_HISTORIES          = "1e6"        # TOTAL NUMBER OF HISTORIES, OR SIMULATION TIME IN SECONDS IF VALUE < 100000
                                    # (Simulating only 10% of the expected exposure for testing! ORIGINAL: 1.7e11)
                                    # Stored as string to preserve exact scientific notation for MC-GPU
RANDOM_SEED          = 131415990    # RANDOM SEED (ranecu PRNG)
GPU_NUMBER           = 10           # GPU NUMBER TO USE WHEN MPI IS NOT USED, OR TO BE AVOIDED IN MPI RUNS
GPU_THREADS_PER_BLOCK = 256         # GPU THREADS PER CUDA BLOCK (multiple of 32)
HISTORIES_PER_THREAD = 25           # SIMULATED HISTORIES PER GPU THREAD


# [SECTION SOURCE v.2016-12-02]

SPECTRUM_FILE        = "spectrum/monocromatic.spc"  # X-RAY ENERGY SPECTRUM FILE

SOURCE_X             =   0.0        # SOURCE POSITION: X (chest-to-nipple) [cm]
SOURCE_Y             =  65.0        # SOURCE POSITION: Y (right-to-left) [cm]
SOURCE_Z             = -53.130      # SOURCE POSITION: Z (caudal-to-cranial) [cm]

SOURCE_DIR_U         =  0.0         # SOURCE DIRECTION COSINES: U
SOURCE_DIR_V         = -1.0         # SOURCE DIRECTION COSINES: V
SOURCE_DIR_W         =  0.0         # SOURCE DIRECTION COSINES: W

BEAM_APERTURE_AZIMUTHAL = 54.0      # TOTAL AZIMUTHAL (WIDTH, X) APERTURE OF THE FAN BEAM [degrees]
                                    # (==> 2/3 original angle of 11.203; input negative to auto-cover detector)
BEAM_APERTURE_POLAR     =  1.5      # TOTAL POLAR (HEIGHT, Z) APERTURE OF THE FAN BEAM [degrees]

EULER_ALPHA          =   0.0        # EULER ANGLES (RzRyRz): alpha — rotate beam from default Y=0, normal=(0,-1,0)
EULER_BETA           =   0.0        # EULER ANGLES (RzRyRz): beta
EULER_GAMMA          = 180.0        # EULER ANGLES (RzRyRz): gamma

FOCAL_SPOT_FWHM      = 0.0300       # SOURCE GAUSSIAN FOCAL SPOT FWHM [cm]
ANGULAR_BLUR         = 0.0          # ANGULAR BLUR DUE TO MOVEMENT ([exposure_time]*[angular_speed]) [degrees]
                                    # (0.18 for DBT, 0 for FFDM [Mackenzie2017])
COLLIMATE_BEAM       = "NO"         # COLLIMATE BEAM TOWARDS POSITIVE AZIMUTHAL (X) ANGLES ONLY? [YES/NO]
                                    # (ie, cone-beam center aligned with chest wall in mammography)


# [SECTION IMAGE DETECTOR v.2017-06-20]

OUTPUT_NAME          = "monocromatic_results/bern"  # OUTPUT IMAGE FILE NAME

DETECTOR_NX          = 1100         # NUMBER OF PIXELS IN THE IMAGE: Nx
DETECTOR_NZ          = 10           # NUMBER OF PIXELS IN THE IMAGE: Nz

DETECTOR_WIDTH_X     = 110.0        # IMAGE SIZE width  Dx [cm]
DETECTOR_HEIGHT_Z    =   1.0        # IMAGE SIZE height Dz [cm]

SDD                  = 110.00       # SOURCE-TO-DETECTOR DISTANCE [cm]
                                    # (detector set in front of source, perpendicular to initial direction)

IMAGE_OFFSET_X       = 0.0          # IMAGE OFFSET ON DETECTOR PLANE: width direction [cm]
IMAGE_OFFSET_Z       = 0.0          # IMAGE OFFSET ON DETECTOR PLANE: height direction [cm]
                                    # (by default beam is centered at image center)

DETECTOR_THICKNESS   = 0.0200       # DETECTOR THICKNESS [cm]
DETECTOR_MFP         = 0.004027     # DETECTOR MATERIAL MEAN FREE PATH AT AVERAGE ENERGY [cm]
                                    # (==> MFP(Se, 19.0keV))

DETECTOR_KEDGE_ENERGY   = 12658.0   # DETECTOR K-EDGE ENERGY [eV]
DETECTOR_KFLUOR_ENERGY  = 11223.0   # K-FLUORESCENCE ENERGY [eV]
DETECTOR_KFLUOR_YIELD   =  0.596    # K-FLUORESCENCE YIELD
DETECTOR_KFLUOR_MFP     =  0.00593  # MFP AT FLUORESCENCE ENERGY [cm]

DETECTOR_GAIN        =  0.0         # EFFECTIVE DETECTOR GAIN, W_+- [eV/ehp] (input 0 to report ideal energy fluence)
SWANK_FACTOR         =  1.0         # SWANK FACTOR

ELECTRONIC_NOISE     =  0.0         # ADDITIVE ELECTRONIC NOISE LEVEL (electrons/pixel)

COVER_THICKNESS      =  0.10        # PROTECTIVE COVER THICKNESS (detector+grid) [cm]
                                    # (==> MFP(polystyrene, 19keV))
COVER_MFP            =  1.9616      # MEAN FREE PATH AT AVERAGE ENERGY [cm]

GRID_RATIO           = 0            # ANTISCATTER GRID RATIO [X:1] (enter 0 to disable)
GRID_FREQUENCY       = 0            # ANTISCATTER GRID FREQUENCY [lp/cm]
GRID_STRIP_THICKNESS = 0            # ANTISCATTER GRID STRIP THICKNESS [cm]

GRID_STRIPS_MFP      = 0.00089945   # ANTISCATTER STRIPS MEAN FREE PATH AT AVERAGE ENERGY [cm]
                                    # (==> MFP(lead, 19keV))
GRID_INTERSPACE_MFP  = 1.9616       # ANTISCATTER INTERSPACE MEAN FREE PATH AT AVERAGE ENERGY [cm]
                                    # (==> MFP(polystyrene, 19keV))

GRID_ORIENTATION     = 0            # ORIENTATION 1D FOCUSED ANTISCATTER GRID LINES:
                                    # 0 == STRIPS PERPENDICULAR LATERAL DIRECTION (mammo style)
                                    # 1 == STRIPS PARALLEL LATERAL DIRECTION (DBT style)


# [SECTION TOMOGRAPHIC TRAJECTORY v.2016-12-02]

N_PROJECTIONS              = 360    # NUMBER OF PROJECTIONS (1 disables tomographic mode)
                                    # (==> 1 for mammo only; ==> 25 for mammo + DBT)
SOD                        =  65.0  # SOURCE-TO-ROTATION AXIS DISTANCE [cm]
ANGLE_BETWEEN_PROJECTIONS  =   1.0  # ANGLE BETWEEN PROJECTIONS (360/num_projections for full CT) [degrees]
ROTATION_TO_FIRST_PROJECTION = 0.0  # ANGULAR ROTATION TO FIRST PROJECTION [degrees]
                                    # (useful for DBT; input source direction considered as 0 degrees)

ROT_AXIS_X           = 0.0          # AXIS OF ROTATION: Vx
ROT_AXIS_Y           = 0.0          # AXIS OF ROTATION: Vy
ROT_AXIS_Z           = 1.0          # AXIS OF ROTATION: Vz

TRANSLATION_ALONG_AXIS = 0.0        # TRANSLATION ALONG ROTATION AXIS BETWEEN PROJECTIONS (HELICAL SCAN) [cm]
KEEP_DETECTOR_FIXED  = "NO"         # KEEP DETECTOR FIXED AT 0 DEGREES FOR DBT? [YES/NO]
SIMULATE_BOTH        = "NO"         # SIMULATE BOTH 0 deg PROJECTION AND TOMOGRAPHIC SCAN (WITHOUT GRID)
                                    # WITH 2/3 TOTAL NUM HIST IN 1st PROJ (eg, DBT+mammo)? [YES/NO]


# [SECTION DOSE DEPOSITION v.2012-12-12]

TALLY_MATERIAL_DOSE  = "NO"         # TALLY MATERIAL DOSE? [YES/NO] (disabled for crash isolation)
TALLY_VOXEL_DOSE     = "NO"         # TALLY 3D VOXEL DOSE? [YES/NO] (dose measured separately for each voxel)
DOSE_OUTPUT_FILE     = "mc-gpu_dose.dat"  # OUTPUT VOXEL DOSE FILE NAME
DOSE_ROI_X_MIN       =   1          # VOXEL DOSE ROI: X-index min (must be within phantom Nx=512)
DOSE_ROI_X_MAX       = 512          # VOXEL DOSE ROI: X-index max
DOSE_ROI_Y_MIN       =   1          # VOXEL DOSE ROI: Y-index min (must be within phantom Ny=512)
DOSE_ROI_Y_MAX       = 512          # VOXEL DOSE ROI: Y-index max
DOSE_ROI_Z_MIN       = 470          # VOXEL DOSE ROI: Z-index min
DOSE_ROI_Z_MAX       = 470          # VOXEL DOSE ROI: Z-index max


# [SECTION VOXELIZED GEOMETRY FILE v.2017-07-26]

PHANTOM_FILE         = "./parcel_s1_Dcropped_14012022_1_ct_Dcropped_14012022_1_ct_voxelId.bin"

PHANTOM_OFFSET_X     = -39.0        # OFFSET OF THE VOXEL GEOMETRY X (DEFAULT ORIGIN AT LOWER BACK CORNER) [cm]
PHANTOM_OFFSET_Y     = -39.0        # OFFSET OF THE VOXEL GEOMETRY Y [cm]
PHANTOM_OFFSET_Z     = -53.130      # OFFSET OF THE VOXEL GEOMETRY Z [cm]

PHANTOM_NX           = 512          # NUMBER OF VOXELS X (input 0 to read ASCII format with header; otherwise raw)
PHANTOM_NY           = 512          # NUMBER OF VOXELS Y
PHANTOM_NZ           = 644          # NUMBER OF VOXELS Z

VOXEL_SIZE_X         = 0.15234375           # VOXEL SIZE X [cm]
VOXEL_SIZE_Y         = 0.15234375           # VOXEL SIZE Y [cm]
VOXEL_SIZE_Z         = 0.1649999976158142   # VOXEL SIZE Z [cm]

TREE_NX              = 1            # SIZE OF LOW RESOLUTION VOXELS (BINARY TREE), GIVEN AS POWERS OF TWO
TREE_NY              = 1            # (eg, 2 2 3 = 2^2 x 2^2 x 2^3 = 128 input voxels per low res voxel)
TREE_NZ              = 1            # (0 0 0 disables tree)


# [SECTION MATERIAL FILE LIST v.2009-11-30]
# Each entry: (material_file, density [g/cm3], voxelId)

MATERIALS = [
    ("MC-GPU_materials_bern/Air_0_5_150keV.mcgpu",              0.0012, 0),   #  1
    ("MC-GPU_materials_bern/Lung_1_5_150keV.mcgpu",             0.1527, 1),   #  2
    ("MC-GPU_materials_bern/Lung_2_5_150keV.mcgpu",             0.3527, 2),   #  3
    ("MC-GPU_materials_bern/Lung_3_5_150keV.mcgpu",             0.5527, 3),   #  4
    ("MC-GPU_materials_bern/Lung_4_5_150keV.mcgpu",             0.7527, 4),   #  5
    ("MC-GPU_materials_bern/Lung_5_5_150keV.mcgpu",             0.8800, 5),   #  6
    ("MC-GPU_materials_bern/AT_AG_SI1_6_5_150keV.mcgpu",        0.9269, 6),   #  7
    ("MC-GPU_materials_bern/AT_AG_SI2_7_5_150keV.mcgpu",        0.9574, 7),   #  8
    ("MC-GPU_materials_bern/AT_AG_SI3_8_5_150keV.mcgpu",        0.9843, 8),   #  9
    ("MC-GPU_materials_bern/AT_AG_SI4_9_5_150keV.mcgpu",        1.0112, 9),   # 10
    ("MC-GPU_materials_bern/AT_AG_SI5_10_5_150keV.mcgpu",       1.0295, 10),  # 11
    ("MC-GPU_materials_bern/SoftTissus_11_5_150keV.mcgpu",      1.0616, 11),  # 12
    ("MC-GPU_materials_bern/ConnectiveTissue_12_5_150keV.mcgpu",1.1199, 12),  # 13
    ("MC-GPU_materials_bern/Marrow_Bone01_13_5_150keV.mcgpu",   1.1112, 13),  # 14
    ("MC-GPU_materials_bern/Marrow_Bone02_14_5_150keV.mcgpu",   1.1645, 14),  # 15
    ("MC-GPU_materials_bern/Marrow_Bone03_15_5_150keV.mcgpu",   1.2237, 15),  # 16
    ("MC-GPU_materials_bern/Marrow_Bone04_16_5_150keV.mcgpu",   1.2830, 16),  # 17
    ("MC-GPU_materials_bern/Marrow_Bone05_17_5_150keV.mcgpu",   1.3422, 17),  # 18
    ("MC-GPU_materials_bern/Marrow_Bone06_18_5_150keV.mcgpu",   1.4014, 18),  # 19
    ("MC-GPU_materials_bern/Marrow_Bone07_19_5_150keV.mcgpu",   1.4607, 19),  # 20
    ("MC-GPU_materials_bern/Marrow_Bone08_20_5_150keV.mcgpu",   1.5199, 20),  # 21
    ("MC-GPU_materials_bern/Marrow_Bone09_21_5_150keV.mcgpu",   1.5791, 21),  # 22
    ("MC-GPU_materials_bern/Marrow_Bone10_22_5_150keV.mcgpu",   1.6384, 22),  # 23
    ("MC-GPU_materials_bern/Marrow_Bone11_23_5_150keV.mcgpu",   1.6976, 23),  # 24
    ("MC-GPU_materials_bern/Marrow_Bone12_24_5_150keV.mcgpu",   1.7569, 24),  # 25
    ("MC-GPU_materials_bern/Marrow_Bone13_25_5_150keV.mcgpu",   1.8161, 25),  # 26
    ("MC-GPU_materials_bern/Marrow_Bone14_26_5_150keV.mcgpu",   1.8753, 26),  # 27
    ("MC-GPU_materials_bern/Marrow_Bone15_27_5_150keV.mcgpu",   1.9464, 27),  # 28
    ("MC-GPU_materials_bern/AmalgamTooth_28_5_150keV.mcgpu",    2.0881, 28),  # 29
    ("MC-GPU_materials_bern/AmalgamTooth_29_5_150keV.mcgpu",    2.2851, 29),  # 30
    ("MC-GPU_materials_bern/MetallImplants_30_5_150keV.mcgpu",  2.4821, 30),  # 31
    ("MC-GPU_materials_bern/MetallImplants_31_5_150keV.mcgpu",  2.6821, 31),  # 32
    ("MC-GPU_materials_bern/MetallImplants_32_5_150keV.mcgpu",  2.7910, 32),  # 33
    ("MC-GPU_materials_bern/MetallImplants_33_5_150keV.mcgpu",  2.8000, 33),  # 34
    ("MC-GPU_materials_bern/MetallImplants_34_5_150keV.mcgpu",  2.8000, 34),  # 35
]
