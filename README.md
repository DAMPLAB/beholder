# Beholder
---
## Available Commands
All commands are run via the poetry environment at the top level of the code repository, e.g. `poetry run python cli.py <command> <arguments>`

The commands are presented in the order in which they should be run:
- Convert the input ND2 files to tiffs. (`convert-nd2-to-tiffs`)
- Perform segmentation and labeling to the tiffs (`segmentation`)
- Upload this data to our remote server. (`s3-sync-upload`)
- (If needed) Download the results to other locations (`s3-sync-download`)

### The entire Beholder pipeline.
```
Example Command: 
$ poetry run python cli.py beholder <PATH TO RUNLIST HERE> --nd2-directory 
<PATH TO DIRECTORIES WITH ND2 FILES> --output-directory <PATH TO OUTPUT DIRECTORY> 
    
Performs all stages and actions described in the passed in runlist. This is how
you should utilize beholder if you using it as an end-to-end nd2 to segmented
results pipeline.

The `beholder` command uses a special json file called a runlist that describes
the pipeline stages, their order, and their arguments.

An example runlist:
{
  "input_datasets": [
    "1-SR_1_5_6hPre-C_2h_1mMIPTG_OFF_1hmMIPTG_ON_22hM9_TS_MC1",
    "1-SR_1_9_6hPre-C_1haTc_TS_MC2",
    "1-SR_2_6_6hPre-C_1hNoIPTG_16hIPTG_M9_TS_MC1",
    "1-SR_3_7_12hPreCulture_2h_IPTG_OFF_12hIPTG_ON_M9_TS_MC1",
    "1-SR_4_15_6hPre-C_1haTc_TS_MC1",
    "2-SR_3_13_6hIPTGOFF_16hIPTGOFF_M9_TS_MC1",
    "2-SR_4_8_12hIPTG_ON_M9_TS_MC1001",
    "2-SR_4_15_1hmMIPTGON_15hM9_TS_MC1"
  ],
  "num_observations": [
    15,
    17,
    17,
    30,
    18,
    21,
    17,
    18
  ],
  "stages": [
    "convert_nd2_to_tiffs",
    "segmentation",
    "s3_sync_upload"
  ],
  "settings": []
}

Field by field breakdown:
- input_datasets: A list of what datasets to use, which in the current case
    means ND2 files. These files should all live in the nd2 directory specified
    above.
- num_observations: The number of observations in a dataset. You only need to 
    include this particular field if you are dealing with the DAPI problem,
    which seemingly stems from settings using the microscope.
- stages: The stages that will be run, and in which order. A stage is a pipeline
    phase that's output will be passed into the next subsequent phase. If you
    turned files into tiffs for instance and the next stage is segmentation,
    the segmentation function will operate on those tiffs.
-  settings: Currently unused, but in the future will be used for pipeline 
    arguments that are non-sequential.
```


### convert-nd2-to-tiffs
<p align="center">
  <img src="https://cidar-screencast-bucket.s3.amazonaws.com/beholder-conversion.gif">
</p>

```
Example Command: 
$ poetry run python cli.py convert-nd2-to-tiffs 
--input-directory <ABSOLUTE-PATH-TO-INPUT-DIRECTORY> 
--output-directory <ABSOLUTE-PATH-TO-OUTPUT-DIRECTORY>

Available Arguments:
    --input-directory: Which directory to get the ND2 files from.
    --output-directory: Which directory to output the converted files to.
    --logging: Whether to log the status of an ongoing conversion to the terminal.

Collects all of the available ND2 files in the passed in input directory and 
asks which one(s) the user would like to convert into tiff files. If all is 
selected the conversion process will be done on all. This process is required 
before running the segmentation process.
```    
### segmentation

```
$ poetry run python cli.py segmentation --input-directory <ABSOLUTE-PATH-TO-INPUT-DIRECTORY>


Available Arguments:
    --input-directory: Which directory to get the tiff files from AND where to place the files.
    --render-videos (Default: True) : Whether to generate segmentation visualizations.
    --logging: Whether to log the status of an ongoing segmentation to the terminal.
    --filter-criteria: If set, filters the input datasets based on the number of channels.

Collects all of the available tiff file directories in the passed in input directory and 
asks which one(s) the user would like to perform segmentation on. If all is 
selected the conversion process will be done on all.
```    

### s3-sync-upload

```
$ poetry run python cli.py s3-sync-upload --input-directory <ABSOLUTE-PATH-TO-INPUT-DIRECTORY> --output-bucket <S3_BUCKET>


Available Arguments:
    --input-directory: Which directory to sync to S3.
    --output-bucket: Which bucket to upload to.
    --results_only: Whether to sync only result files (Excludes tiffs and nd2s)

Performs a sync operation between the passed in local directory and the one on AWS.
This should never delete files but will overwrite them if they have the same name 
and you have a newer copy local to your machine.
```    

### s3-sync-download

```
$ poetry run python cli.py s3-sync-download --output-directory <ABSOLUTE-PATH-TO-OUTPUT-DIRECTORY> --input-bucket <S3_BUCKET>


Available Arguments:
    --output-directory: Which directory to sync with the S3 remote.
    --input-bucket: Which bucket to sync with.
    --results_only: Whether to sync only result files (Excludes tiffs and nd2s)

Performs a sync operation between the passed in local directory and the one on AWS.
This should just pull down files to your local machine.
```    



# Installation
##Install the Following:

##[Poetry](https://python-poetry.org/docs/#installation) (Python Environment Resolution)

[ImageJ](https://imagej.net/Fiji/Downloads) (Opensource Scientific Image Viewer)
---
Python Installation Instructions
```
$ poetry config virtualenvs.in-project true
$ poetry install # This will hang on the OpenCV Installation due to a current bug. This should go away.
$ poetry run pip install --upgrade pip 
$ poetry install # This should now install everything.
```
