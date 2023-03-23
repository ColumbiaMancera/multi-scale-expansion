# Multi Scale Expansion
Repository to create framework for multi-scale computer vision models in tensorflow. <br>
![image](https://img.shields.io/pypi/l/tensorflow)
![image](https://img.shields.io/github/issues/ColumbiaMancera/multi-scale-expansion)
![Build Status](https://github.com/ColumbiaMancera/multi-scale-expansion/actions/workflows/build.yml/badge.svg)
[![codecov](https://codecov.io/gh/ColumbiaMancera/multi-scale-expansion/branch/main/graph/badge.svg)](https://codecov.io/gh/ColumbiaMancera/multi-scale-expansion)

## Overview
`multi-scale-expansion` is a library for automating the creation of a multi-scale computer vision framework. The user would provide their untrained model with a tensorflow basis, as well as their data, and the library would expand it to a structure where there'd be a model per zoom level for a particular classification task.
