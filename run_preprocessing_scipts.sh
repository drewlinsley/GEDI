#!/bin/sh

python preprocessing_scripts/extract_GEDI_and_prepare_tf_records_tvt.py
#python preprocessing_scripts/extract_frames_from_GEDI.py  # extracts tiffs -> pngs
#python preprocessing_scripts/prepare_tf_records.py  # pngs -> tfrecords
