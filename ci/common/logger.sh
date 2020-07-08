#!/bin/bash


# Logger function for build status output
function logger() {
  ELAPSED_TIME=$(( $(date +%s)-START_TIME ));
  PRINT_TIME=$(printf '%dh:%dm:%ds\n' $((ELAPSED_TIME/3600)) $((ELAPSED_TIME%3600/60)) $((ELAPSED_TIME%60)));
  echo -e "\n${PRINT_TIME} >>>> $* <<<<\n"
}
