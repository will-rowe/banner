<div align="center">
    <img src="https://raw.githubusercontent.com/will-rowe/hulk/master/paper/img/misc/hulk-logo-with-text.png" alt="hulk-logo" width="250">
    <h3><a style="color:#9900FF">BANNER</a></h3>
    <hr>
    <a href="https://travis-ci.org/will-rowe/banner"><img src="https://travis-ci.org/will-rowe/banner.svg?branch=master" alt="travis"></a>
    <a href="https://github.com/will-rowe/master/blob/master/LICENSE"><img src="https://img.shields.io/badge/license-MIT-orange.svg" alt="License"></a>
    <a href="https://zenodo.org/badge/latestdoi/144629592"><img src="https://zenodo.org/badge/144629592.svg" alt="DOI"></a>
</div>

***

```
BANNER is still under development - features and improvements are being added, so please check back soon.
```

***

## Overview

`BANNER` is a tool that lives inside [HULK](https://github.com/will-rowe/hulk) and aims to make sense of **hulk sketches**. At the moment, it trains a [Random Forest Classifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) using a set of labelled **hulk sketches**. It can then use this model to predict the label of microbiomes as they are sketches by ``HULK``.

For example, you could train `BANNER` using a set of microbiomes from patients that either have or haven't received antibiotic treatment. You can then use `BANNER` to predict whether a new microbiome sample exhibits signs of antibiotic dysbiosis. I will post more information and examples soon...

## Installation

### Bioconda

```
conda install banner
```

> note: if using Conda make sure you have added the [Bioconda](https://bioconda.github.io/) channel first

#### Pip

```
git clone https://github.com/will-rowe/banner && cd $_
pip install -r requirements.txt .
```

## Quick Start

`BANNER` is called by typing **banner**, followed by the subcommand you wish to run. There are two main subcommands: **train** and **predict**. This quick start will show you how to get things running but it is recommended to follow the [HULK documentation](http://hulk-documentation.readthedocs.io/en/latest/?badge=latest).

```bash
# Train a random forest classifier
banner train -m hulk-banner-matrix.csv -o banner.rfc

# Predict the label for a hulk sketch
hulk sketch -f mystery-sample.fastq --stream -p 8 | banner predict -m banner.rfc
```


## Notes

* only supports 2 labels at the moment

* there is very limited checking and not many unit tests...
