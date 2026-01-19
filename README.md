# INPUT TEXT DATA CLEANING PIPELINE
To build:
- install docker on system (https://docs.docker.com/engine/install/)
- opon terminal window
- cd <DOWNLOADED_GIT_REPO_DIRECTORY>:
- add data to folder 'data'
- docker build command (this will build an image with all files and dependencies):
    > docker build -t YOUR_PIPELINE_APPNAME:V# .
- docker run command:
    > docker run YOUR_PIPELINE_APPNAME:V#
- find .jsonl cleaned ouput in image 'result' folder
- find cleaning reports for all chunks in result/report<yyyymmdd> 

## FILES
- DockerFile - run in docker to initiate a container.
- environment.yml - .yml used by docker to install dependencies.
- run.py - top level file to run preprocessing.

## FOLDERS
- /data - place processable .jsonl items in here.
- /result  - processed .jsonl will appear in here.
	-/report<yyyymmdd> - each chunks .jsonl report stored here.
- /src - processing scripts here.

## FILES IN SRC

## read_jsonl.py - Utilities to read and chunk JSONL (newline-delimited JSON) files.

### Functions:
- read_jsonl_with_pandas(path): load into a pandas DataFrame (if pandas installed).
- save_working(path, pd, suff) # saves current version (<path>, <pandas dataFrame>, <suffice for file>)
- check_save_integrity(name, df, path) # optional step to make sure saved file can be save and loaded without data lose
- read(sample, chunks=None) # top level command to read datafile and chunk for processing if needed.

## clean_text.py - A configurable, pipeline to clean text lines before feeding to an LLM.

Features:
### First step so that they can be processed.
- Unicode normalization (NFC)

### Handle foreign language
- translate non-English language
- Remove non-English language

### Remove non usables
- Remove HTML entities
- Collapse_whitespace
- Remove control characters
- Remove non-printable characters
- Shrink repeated characters
- Remove leading and trailing
- Remove code
- fix 'mojibake' (text nonsense)
- Spell checking (english)
    
### Anonomise all person specific info
- Replace Personal Nouns using Natural language processing.
- Replace common profanity 
- Replace IP addresses
- Replace UUIDs
- Replace MAC Addresses
- Replace Timestamps
- Replace specific filepaths
- Replace Social security numbers
- Replace Credit Card data
- Replace Personal Identifiying Info (ie. phone, email, URL)
- Remove Personal Identifyin Info completely
- Replace Currency values

### Remove text they don't meet criteria
- Remove word repetition and set a minimum distance between word token repeats to not delete them.
- Minimum length of string before deleting it
- Maximum number of character for a token
- Maximum number of tokens per entry

### Usage:
from clean_text import Cleaner
proc = Cleaner()
cleaned = proc.clean(raw_lines)

or 

If input is chunks:
from clean_text import Cleaner, clean_chunk
proc = Cleaner()
clean_chunk(pd.DataFrame,proc.cfg)



