# Gridless 3D Recovery of Image Sources from Room Impulse Responses

![alt text](sfwb.gif)

**Update : see below for room geometry estimation**

Adapted Sliding Frank-Wolfe algorithm for 3D image source recovery from room impulse responses. 
The algorithm is described in our paper [Gridless 3D Recovery of Image Sources from Room Impulse Responses](https://hal.archives-ouvertes.fr/hal-03763838v2).
The version of the code used in the paper for image source localization is available in the main branch at commit [1356302](https://github.com/Sprunckt/acoustic-sfw/tree/135630234a8aa16a229b3fcb8500be87c8770e8c).

Executed on python 3.8.10.

**Package versions used:**
    
```matplotlib==3.4.3
numpy==1.20.3
optimparallel==0.1.2
pandas==1.3.4
pyroomacoustics==0.6.0
scikit_learn==1.0.1
scipy==1.7.2
```

Note : A minor bug was later fixed at commit 
[ed6a084](https://github.com/Sprunckt/acoustic-sfw/tree/ed6a084bcfe4791c5ca21fbb51872413f200cd7b)
with a negligible impact on the reported results. A bug in the case where the separation constraints in room generation 
were different (not tested in the paper) has also been fixed and the most recent version of the code should be used.
# I - Image-source localization

## 1 - Basic intructions

To launch a given script, navigate to the project directory and run the script from the command line as a module.

**Example :**
``~/sfw$ python3 -m src.test_sfw``

To launch the automated tests (should run on the main branch): 

``~/sfw$ python3 -m unittest discover tests``

## 2 - Advanced usage

### Launching one or multiple source reconstructions
Simulations can be launched through the apply_sfw.py script, either directly or in command line:

``~/sfw$ python3 -m src.apply_sfw [--path=] [--exp=] [--exp_path]``

**Arguments :**
* path: directory where the results will be saved, should contain a parameters.json configuration file which specifies
the configuration of the experiments
* exp_path: directory containing the room parameters in .json files
* exp: two integers separated by a comma that give the range of experience IDs that will be considered in the folder. Ex : 4,9 will apply SFW to the experiences 4 to 8.

The format of the configuration files and the scripts than can be used to create them are detailed below.

### Creating a configuration file for the rooms

A room file should respect the name "exp_i_param.json" where i is replaced by the room id number. There should be no gaps in the ids in order to use apply_sfw on the whole sequence.
The room file can either be created manually or the script sfw_experiment.py can be used to create random rooms. 
This last case is covered in the next section (recommanded to get a file template that can be modified afterwards).
Some parameters given can be ignored depending on a specific experiment configuration file. 

**Fields:**
* src_pos: array containing the source coordinates
* room_dim: array containing the room dimensions
* origin: location of the center of the microphone array in the room
* absorptions: dictionary containing the walls' absorption coefficients. Example: {"north": 0.07, "south": 0.10, "east": 0.28, "west": 0.16, "floor": 0.15, "ceiling": 0.02}
* rotation_mic (optional): array containing 3 angles for a rotation that will be applied to the microphone antenna before simulations in order to randomize the orientation of the antenna
and shuffle the coordinate referential. 

File content example: 
```{"src_pos": [2.3, 7.7, 3.1], "room_dim": [5.1, 9.6, 4.9], "origin": [1.5, 8., 1.6], "rotation_mic": [5., 56., -44.], "absorptions": {"north": 0.147, "south": 0.282, "east": 0.146, "west": 0.13, "floor": 0.029, "ceiling": 0.025}}```
### Using experiment_sfw.py to create random rooms

To use sfw_experiment.py, a parameters.json configuration file must be placed in the directory that will contain the room files.
This configuration file can be created by launching the script create_conf_file.py after editing the target path in the script, 
and changing the boolean "room_param" set to True. The number of generated rooms can be modified in the file.
The absorption coefficients are chosen uniformly in [0.01, 0.3].

**Fields:**
* xlim, ylim, zlim: set lower and upper bounds for the room length in each direction (picked using a uniform law)
* mic_size: scale factor applied to the radius of the eigenmike spherical antenna
* src_wall_sep, mic_wall_sep: minimal distance of the source/microphone antenna center to the walls (might cause a crash when the room is too small)
* z_src, z_mic: additional constrain to set a fixed z coordinate for the source/microphone antenna center (set to None to keep it random)

### Simulations configuration files

The folder that is chosen to contain the reconstruction results must contain a parameters.json configuration file that 
can be created manually or by using create_conf_file.py with the boolean room_param set to False. In this last case a basic
configuration file is created and can be manually improved/modified.


**Fields:**
* domain: the type of reconstruction considered: "time", "frequency" or "deconvolution" (the default being time domain)
* normalization: integer index giving the normalization used (0 for no normalization)
* max_order: the maximum order of the image sources generated by the simulation
* fs: sampling frequency in Hz
* fc: cutting frequency in Hz, should be < fs
* cutoff: maximum time in the RIR (in s)
* lambda: TV norm regularization parameter
* grid_search: initialization method for spike finding step. Supported: "full", "rough", "naive", "decimation". full and rough both do a gradient descent at each grid point before keeping the best value (requires a very light search grid), naive uses the argmax on the grid as initialization. "decimation" operates as "rough", but works on decimated signals and require smaller grids for high sampling frequency/array size
* min_norm: minimal distance (in m) to the origin of a reconstructed source considered when using the "rough" initialization method
* ideal: use the ideal sinc operator if True, otherwise use approximate pyroom acoustics simulations
* spherical_search: if True, find the time samples maximizing the square of the residual, center the grid at the corresponding microphone(s) and scale the previously generated grid by the expected distance to the image source.
* nmic: number of microphones retained for the grid generated with spherical_search=True (concatenation of every scaled and recentered grid)
* dr, dphi, rmin, rmax: parametrize the spherical grid that will be used for intialization (dphi is in degrees). rmin=rmax=1 corresponds to a unit sphere that can be efficiently used with spherical_search=True (see above)
* multiple_spheres: specify a number of spheres to be added around the radius rmin with spacing dr to generate a grid. Should be used with spherical_search=True, rmin=rmax=1, dr=the wanted spacing. Generates 2*multiple_spheres+1 spheres.
* mic_size: scaling factor for the microphone antenna
* psnr: Peak Signal to Noise Ratio (db)
* max_iter: maximal number of iterations
* use_absorption: if True, use the absorption coefficients given in the room file, otherwise overwrite them with identical, highly reflecting walls
* deletion_tol: amplitude threshold under which a spike is considered negligible and is deleted
* start_cb: dictionary containing the parameters used to control the time segmentation of the RIR
Example: {"n_cut":10, "swap_frequency":20, "swap_factor":0.3, "method":"time"}. Cut the RIR in 10, extend to the next segment if 20 iterations have passed or
if the residual has been reduced by a factor 0.3. Setting n_cut to 0 gets the default behavior (no segmentation). Two methods are supported: "time" which segments linearly the interval, and "energy" which does an adaptative segmentation based on the square of the signals.
* normalization: int which specify one of the available normalization for the linear operator, 0 for default. Some features might not be implemented for other normalizations.
* end_tol: amplitude threshold applied at algorithm stop (applied before and after sliding step if using the single slide method)
* reverse_coordinates: boolean to determine if the inverse rotation should be applied after reconstruction (now 
defaults to false, meaning the ground truth is expressed in the room referential and the reconstructed sources are in the array
 referential)

**Additional argument "slide_opt" to control the sliding step:**

Should be a dictionary containing the options for the sliding step. If None: perform a full sliding
        at each step. Else: should contain a key "method", the associated value being in ["slide_once", "no_slide",
        "freeze"].
Behavior: 
* "slide_once" : skip the sliding step and perform a single sliding at the end
* "no_slide" : completely skip the sliding step
* "freeze" : the additional key "freeze_step" should be added. Check each spike every
        "freeze_step" iterations. If the spike has not moved sufficiently since the last check, the spike is frozen and
        is not allowed to slide in the next iterations.
        Additional option in that case: the key "resliding_step" allows for an additional periodic sliding step to be applied on
        every spike (including the frozen ones).

File content example that can be modified for different use cases: 
```{"domain":"time", "max_order": 20, "fs": 16000, "fc": 16000, "rmin": 1.0, "rmax": 1.0, "dr": 0.05,"dphi": 5, "normalization": 0, "lambda": 3e-5, "ideal": true, "spherical_search": true, "mic_size": 2.0, "max_iter": 1500, "use_absorption": true, "cutoff":0.05, "start_cb":{"n_cut":10, "swap_frequency":20, "swap_factor":0.3, "method":"time"}, "slide_opt":{"method": "slide_once"}, "deletion_tol": 0.01, "end_tol":0.1, "grid_search": "naive", "opt_param":{"roughgtol":1e-4, "roughmaxiter": 100, "gtol": 1e-7}, "multiple_spheres":1, "nmic":8}```

Description: applies the noiseless time domain reconstruction of image sources, with sampling frequency 16000Hz, 2*standard array diameter.
Corresponding initialization method: find an approximate time of arrival for each microphone, keep the 8 best values. 
For each of the 8 times of arrivals selected, generate 3 concentric spheres centered around the corresponding microphone and 
separated by 5cm from each other. The argmax of the certificate eta over the reunion of these grids is used as an initial guess.
Only apply a single sliding step at the end of the algorithm. Note that some arguments are not used here (e.g "roughgtol", "rmax").

# II - Cuboid room geometry estimation

This part of the code is used to estimate the geometric configuration of a cuboid room from the estimated image-source locations.
The functions are contained in src/tools/geometry_reconstruction.py.
The full procedure is detailed in our paper [Fully Reversing the Shoebox Image Source Method: From Impulse Responses to Room Parameters](https://hal.science/hal-04567514).

## 1 - Orientation estimation

The orientation of the room can be recovered by using the RotationFitter class. Assuming that the estimated source
positions are stored in ```recons_pos```, the orientation can be estimated as follows:

```python
fitter = KernelFitter(image_pos=recons_pos)  # input the estimated source positions in the microphone array basis
bandwidth_values3d = [0.01, 0.005, 0.0005]  # 3d bandwitdh values
bandwidth_values2d = [0.01, 0.005, 0.0005]  # 2d bandwitdh values

basis = fitter.fit(gridparam=1., niter=10000, tol=1e-10, bvalues3d=bandwidth_values3d,
                   bvalues2d=bandwidth_values2d, verbose=True, plot=False)
```

```basis``` is a 3x3 rotation matrix that can be used to change the coordinate system to a referential frame of the room.


## 2 - Dimension estimation

Once a basis has been estimated, the geometric information can be recovered by calling ```find_dimensions_ns```
(where ```reconstr_ampl``` contains the estimated source amplitudes):


```python
(dist_found, estimated_src_pos, estimated_src_ampl,
 order1_ampl, order1_found) = find_dimensions_ns(recons_pos, basis=basis,
                                                      amplitudes=reconstr_ampl,
                                                      fusion=True, cone_width=5)
```

* ```dist_found``` is the estimated distance between the original source and the walls (shape (2, 3))
* ```estimated_src_pos``` is the estimated source position in the array referential (shape (3,))
* ```estimated_src_ampl``` is the estimated source amplitude (```1 - src_ampl**2``` gives the absorption coefficient)
* ```order1_ampl``` is the estimated order 1 source amplitudes (shape (3, 2))
* ```order1_pos``` is the estimated order 1 source positions (shape (6, 3))

The whole room configuration can then be estimated using these values.