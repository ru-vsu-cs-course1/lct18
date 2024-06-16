#!/bin/bash

CD=$(dirname "$(readlink -f "$0")")  # "
if [[ -e "$CD/_props.sh" ]]; then
  . "$CD/_props.sh"
fi

SIZE_PARAMS=''


cd "$CD"

for IN_FILE in $(ls -1 ./data/crop/*_0000.tif); do
  if [[ ! -e "${IN_FILE%.*}.png" || \
        ! -e "${IN_FILE%.*}.r.png" || ! -e "${IN_FILE%.*}.g.png" || ! -e "${IN_FILE%.*}.b.png" || \
        ! -e "${IN_FILE%.*}.nir.png" || ! -e "${IN_FILE%.*}.gray.png" ]]; then
    echo "Converting $IN_FILE"
    python ./test1_to_png.py "$IN_FILE" $SIZE_PARAMS -c all
  fi
done
