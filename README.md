# streg


# Compiling the code

Run the following commands to compile the code (OpenCV is required)
1. `cd build`
1. `./builder.sh`
1. If compiling is successful, you an executable file named `streg` should be created


# Running the code

The code runs as below with a total of four parameters; the last two are optional.
`./streg <input_video_file> <part_type> <*optional:* length_of_chunks> <*optional:* save_before_registration>`
- **parameter 1** *input_video_file*: this is a video file that contains only a facial part (left eye, right eye or mouth).
- **parameter 2** *part_type*: This parameter determines which facial part will be registered; it must be set to either `leye`, or `reye` or `mouth`.
- **parameter 3** *length_of_chunks `T`*: The input file is typically divided to `T`-second chunks before being registered. This is necessary, because small registration errors accumulate to large values over time, therefore we need to periodically reset. This parameter (i.e., `T`) determines how long these chunks will be. The default is `T=3` (seconds).
- **parameter 4** *save_before_registration*: A binary parameter (0 or 1) that determines whether the before registration will be saved. The default is 0, but can be set to 1 for inspection.


## Example commands

- `./streg vid01_leye.avi leye ./output `
- `./streg vid01_reye.avi reye ./output `
- `./streg vid01_mouth.avi mouth ./output `

If you want to store the "before registration" videos, you can run the following code

- `./streg vid01_leye.avi leye ./output 3 1`
- `./streg vid01_reye.avi reye ./output 3 1`
- `./streg vid01_mouth.avi mouth ./output 3 1`


# Outputs

## Output 1: Registered videos
The primary output is a set of video files named like below. For example, if the input video file is named vid1.mp4 and we are registering the left eye (`leye`), then the output with the registered videos will be a set of videos as
```
- vid1-leye-0001.avi
- vid1-leye-0002.avi
- vid1-leye-0003.avi
...
```
As mentioned in section Running The Code, we register `T`-second chunks, so each of the videos above is of length `T` seconds.

## Output 2: Indicator of success
Alongside with each chunk like `vid1-leye-0001.avi`, we produce a text file named `vid1-leye-0001.success` that indicates whether or not each frame has been successfully registered. The `t`th line of this file is a single number indicates whether or not the `t`th frame in the file `vid1-leye-0001.avi` is correctly registered. Specifically, the value at each line is one of the four below:

- `0`: the registration of this frame to the previous frame has failed
- `1`: this frame has been successfully registered to the previous frame
- `2`: this frame has been successfully registered to the previous frame after correction
- `3`: this frame is the new reference frame; typically happens after previous frames failed to be registered



