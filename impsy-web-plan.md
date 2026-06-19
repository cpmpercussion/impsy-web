The idea of this repo is to create a "web" version of IMPSY.

The IMPSY project is a platform for creating intelligent musical instruments. The project is centred on a python application which helps collect musical interaction data, train MDRNN models, and interact with them in live performance by facilitating MIDI mappings with the model inputs and outputs.

The Python software (../impsy/) should be considered the canonical interaction concept and the source for how IMPSY applications should work. We have already built an auv3 plugin and mac/iOS app which is in ../impsy-auv3/

This version is going to be a web application that runs impsy .tflite models on the user's web browser. impsy-web will communicate via web-midi with other local MIDI software.

The interface for impsy-web should be aligned with impsy-auv3 and made as close as possible to that product while remaining idiomatic to the web framework used.

The first task is to research feasibility, then investigate any design choices/tradeoffs, then implement.
