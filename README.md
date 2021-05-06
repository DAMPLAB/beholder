# Beholder
---
## Available Commands
All commands are run via the poetry environment at the top level of the code repository, e.g. `poetry run python cli.py <command> <arguments>`

The commands are presented in the order in which they should be run:
- Convert the input ND2 files to tiffs. (`convert-nd2-to-tiffs`)
- Perform segmentation and labeling to the tiffs (`segmentation`)
- Upload this data to our remote server. (`s3-sync-upload`)
- (If needed) Download the results to other locations (`s3-sync-download`)

### convert-nd2-to-tiffs
<p align="center">
  <img src="https://cidar-screencast-bucket.s3.amazonaws.com/beholder-conversion.gif">
</p>

```
Example Command: 
$ poetry run python cli.py convert-nd2-to-tiffs --input-directory <ABSOLUTE-PATH-TO-INPUT-DIRECTORY> --output-directory <ABSOLUTE-PATH-TO-OUTPUT-DIRECTORY>
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
