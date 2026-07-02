#!/bin/bash

# Copyright © 2019-2023
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# this script uses sv2v and yosys tools to run.
# sv2v: https://github.com/zachjs/sv2v
# yosys: http://www.clifford.at/yosys/

# exit when any command fails
set -e

library=""
sdc_file=""
source=""
top_level=""
dir_list=()
inc_args=""
macro_args=""
no_warnings=1
process="elaborate,netlist,techmap,verilog"

declare -a excluded_warnings=("Resizing cell port")

is_excluded_warning() {
    local warning_text="$1"
    for exclusion in "${excluded_warnings[@]}"; do
        if [[ "$warning_text" == *"$exclusion"* ]]; then
            return $no_warnings
        fi
    done
    return 1
}

checkErrors()
{
    log_file="$1"
    if grep -q "Error: " "$log_file"; then
        echo "Error: found errors during synthesis!"
        exit 1
    fi

    # -W: allow warnings (only hard errors, checked above, fail the run)
    if [ "$no_warnings" -eq 0 ]; then
        return
    fi

    count=0
    while IFS= read -r line; do
        if [[ "$line" == *"Warning:"* ]]; then
            warning_text="${line#Warning: }"
            if ! is_excluded_warning "$warning_text"; then
                count=$(expr $count + 1)
            fi
        fi
    done < $log_file

    if [ "$count" -ne 0 ]; then
        echo "Error: found $count unexpected warnings during synthesis!"
        exit $count
    fi
}

usage() { echo "$0 usage:" && grep " .)\ #" $0; exit 0; }
blackbox_list=()
keep_list=()
[ $# -eq 0 ] && usage
while getopts "c:l:s:t:I:D:P:B:K:Wh" arg; do
    case $arg in
    l) # library
        library=${OPTARG}
        ;;
    c) # SDC constraints
        sdc_file=${OPTARG}
        ;;
    s) # source
        source=${OPTARG}
        ;;
    t) # top-level
        top_level=${OPTARG}
        ;;
    I) # include directory
        dir_list+=(${OPTARG})
        inc_args="$inc_args -I${OPTARG}"
        ;;
    D) # macro definition
        macro_args="$macro_args -D${OPTARG}"
        ;;
    P) # process
        process=${OPTARG}
        ;;
    B) # black-box module (exclude its internals from synthesis, e.g. an SRAM macro)
        blackbox_list+=(${OPTARG})
        ;;
    K) # keep module (mark keep so DCE can't strip it, e.g. a block with a dead output)
        keep_list+=(${OPTARG})
        ;;
    W) # allow warnings
        no_warnings=0
        ;;
    h | *)
      usage
      exit 0
      ;;
  esac
done

{
    # read device library
    if [ -n "$library" ]; then
        echo "read_liberty $library"
    fi

    # read design constraints
    if [ -n "$sdc_file" ]; then
        echo "read_sdc $sdc_file"
    fi

    # read design sources
    for dir in "${dir_list[@]}"
    do
        for file in $(find $dir -maxdepth 1 -name '*.v' -o -name '*.sv' -type f)
        do
            echo "read_verilog -defer -nolatches $macro_args $inc_args -sv $file"
        done
    done
    if [ -n "$source" ]; then
        echo "read_verilog -defer -nolatches $macro_args $inc_args -sv $source"
    fi

    # black-box / keep requested modules.  Elaborate first so parameterized modules
    # exist under their derived $paramod name, then match by wildcard.
    #  -B: blackbox — keep the interface but drop the contents, so synthesis excludes
    #      them from the gate count (compiled SRAM macros quoted from a datasheet).
    #      Matching the base module alone doesn't stick (synth re-derives the
    #      parameterized copy with contents), so this must run after hierarchy.
    #  -K: keep — mark the module so DCE can't strip it; needed to measure a block
    #      whose output isn't observable at the top (e.g. a flag that only feeds sim).
    if [ ${#blackbox_list[@]} -gt 0 ] || [ ${#keep_list[@]} -gt 0 ]; then
        echo "hierarchy -top $top_level"
        for bb in "${blackbox_list[@]}"; do
            echo "blackbox *$bb*"
        done
        for kk in "${keep_list[@]}"; do
            echo "setattr -mod -set keep 1 *$kk*"
        done
    fi

    # elaborate
    if echo "$process" | grep -q "elaborate"; then
        echo "hierarchy -top $top_level"
    fi

    # synthesize design
    if echo "$process" | grep -q "synthesis"; then
        echo "synth -top $top_level"
    fi

    # convert to netlist
    if echo "$process" | grep -q "netlist"; then
        echo "proc; opt"
    fi

    # convert to gate logic
    if echo "$process" | grep -q "techmap"; then
        echo "techmap; opt"
    fi

    # write synthesized design
    if echo "$process" | grep -q "verilog"; then
        echo "write_verilog synth.v"
    fi

    # Generate a summary report
    echo "stat"
} > synth.ys

yosys -l yosys.log -s synth.ys

checkErrors yosys.log
