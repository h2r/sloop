#!/bin/bash

cd sloop/oopomdp/experiments
mkdir resources
cd resources

fileid="1O_4IyJhFJjSuFHxewhwb07Ljrp_hpBXH"
filename="models.tar.gz"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
rm -rf cookie

tar xzvf $filename
rm -rf $filename
