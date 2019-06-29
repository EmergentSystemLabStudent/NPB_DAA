#!/bin/bash

ls DATA | sed -e "s/.txt//g" > files.txt
