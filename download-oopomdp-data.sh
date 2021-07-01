#!/bin/bash

cd sloop/oopomdp/experiments
mkdir resources
cd resources

# WARNING: THIS ID IS CURRENTLY THE OLD ID
# WARNING: THIS ID DOESN'T DOWNLOAD A VALID TAR FILE
fileid="1dGYJBHFCF8NR8Sb7ZIcETexP2F_ADRts"
filename="models.tar.gz"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
rm -rf cookie

tar xzvf $filename
rm -rf $filename
