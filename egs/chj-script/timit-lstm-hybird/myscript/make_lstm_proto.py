#!/usr/bin/python

# Copyright 2014  Brno University of Technology (author: Karel Vesely)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

# Generated Nnet prototype, to be initialized by 'nnet-initialize'.

import math, random, sys
from optparse import OptionParser

###
### Parse options
###
usage="%prog [options] <feat-dim> <num-leaves> <num-lstm-layers> <num-lstm-cell> <num-lstm-stream> <num-recurrent-neurons>  >nnet-proto-file"
parser = OptionParser(usage)

parser.add_option('--no-softmax', dest='with_softmax', 
                   help='Do not put <SoftMax> in the prototype [default: %default]', 
                   default=True, action='store_false');
parser.add_option('--lstm-param-scale', dest='lstm_param_scale', 
                   help='Factor to rescale Normal distriburtion for initalizing weight matrices in LSTM layers [default: %default]', 
                   default=0.01, type='float');
parser.add_option('--param-stddev-factor', dest='param_stddev_factor', 
                   help='Factor to rescale Normal distriburtion for initalizing weight matrices [default: %default]', 
                   default=0.04, type='float');

(o,args) = parser.parse_args()
if len(args) != 6 : 
  parser.print_help()
  sys.exit(1)
  
(feat_dim, num_leaves, num_lstm_layers, num_cells, num_streams, num_recurrent_neurons) = map(int,args);
### End parse options 


# Check
assert(feat_dim > 0)
assert(num_leaves > 0)
assert(num_lstm_layers >= 0)
assert(num_cells > 0)
assert(num_streams > 0)
assert(num_recurrent_neurons > 0)

###
### Print prototype of the network
###

# Only last layer (logistic regression)
if num_lstm_layers == 0:
  print "<NnetProto>"
  print "<AffineTransform> <InputDim> %d <OutputDim> %d <BiasMean> %f <BiasRange> %f <ParamStddev> %f" % \
        (feat_dim, num_leaves, 0.0, 0.0, o.param_stddev_factor)
  if o.with_softmax:
    print "<Softmax> <InputDim> %d <OutputDim> %d" % (num_leaves, num_leaves)
  print "</NnetProto>"
  # We are done!
  sys.exit(0)

# Assuming we have >0 hidden layers
assert(num_lstm_layers > 0)

# Begin the prototype
print "<NnetProto>"

# First LSTM layer
print "<Transmit> <InputDim> %d <OutputDim> %d" % \
	(feat_dim, feat_dim)
print "<LstmProjectedStreams> <InputDim> %d <OutputDim> %d <CellDim> %d <ParamScale> %f <NumStream> %d" % \
	(feat_dim, num_recurrent_neurons, num_cells, o.lstm_param_scale, num_streams)

# Internal LSTM
for i in range(num_lstm_layers-1):
  print "<Transmit> <InputDim> %d <OutputDim> %d" % \
	(num_recurrent_neurons, num_recurrent_neurons)
  print "<LstmProjectedStreams> <InputDim> %d <OutputDim> %d <CellDim> %d <ParamScale> %f <NumStream> %d" % \
	(num_recurrent_neurons, num_recurrent_neurons, num_cells, o.lstm_param_scale, num_streams)

# Last AffineTransform
print "<AffineTransform> <InputDim> %d <OutputDim> %d <BiasMean> %f <BiasRange> %f <ParamStddev> %f" % \
      (num_recurrent_neurons, num_leaves, 0.0, 0.0, o.param_stddev_factor)

# Optionaly append softmax
if o.with_softmax:
  print "<Softmax> <InputDim> %d <OutputDim> %d" % (num_leaves, num_leaves)

# End the prototype
print "</NnetProto>"

# We are done!
sys.exit(0)
