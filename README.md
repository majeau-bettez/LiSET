Lifecycle Screening of Emerging Technology (LiSET) Framework
============================================================

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1257652.svg)](https://doi.org/10.5281/zenodo.1257652)


This repository hosts a clustering tool to facilitate semi-quantitative, relative scoring of competing candidates in terms of multiple lifecycle aspects.


Motivation
----------

The Lifecycle Screening of Emerging Technologies (LiSET) is a framework to systematically and rapidly gain an overview of the environmental hotspots, relative strengths, and potential environmental show-stoppers for a large number of technology candidates. It provides guidance to combine expert judgements, qualitative data, and uncertain quantitative data in a visual map of factors likely to determine the future environmental performance of emerging technologies (a.k.a. lifecycle aspects).

> Hung, C. R., Ellingsen, L. A.-W., Majeau-Bettez, G.(2018). LiSET: a framework for early stage Lifecycle Screening of Emerging Technologies. _Journal of Industrial Ecology_, In Review.

To combine quantitative and semi-quantitative aspects in a consistent visual map, groups of technologies are defined. How could one "objectively" divide a set of 25 competing technologies in three groups based on their relative energy efficiency? LiSET uses clustering algorithms to achieve this.


Demo
-----

Have a look at this demo for a [quick overview of basic functionality](https://github.com/majeau-bettez/LiSET/blob/master/doc/demo_liset_clusters.ipynb).


Key Features
-------------

This module:
* Provides functions directly mirroring the requirements of the LiSET framework;
* Offers clustering based on Jenks Natural Breaks and on K-Means algorithms;
* Plots univariate clusters and lifecycle-aspect heat maps;
* Handles and highlights data gaps to accompany an iterative screening process;
* Accepts data in various formats (lists, Numpy arrays, Pandas series and dataframes).

If this code is useful to you, please cite the LiSET framework by Hung et al., and cite this code with its Digital Object Identifier (DOI).

Use
---

This module has been used in two lifecycle screening studies:

> Ellingsen, L. A.-W., Holland, A., Drillet, J.-F., Peters, W., Eckert, M., Concepcion, C., Ruiz, O., Colin, J.-F., Knipping, E., Pan, Q., Wills, R. G. A. and Majeau-Bettez, G. (2018) ‘Environmental Screening of Electrode Materials for a Rechargeable Aluminum Battery with an AlCl3/EMIMCl Electrolyte’, _Materials_, 11(6). doi: [10.3390/ma11060936](http://dx.doi.org/10.3390/ma11060936).

> Ellingsen, L. A.-W., Hung, C. R., Majeau-Bettez, G., Singh, B., Chen, Z., Whittingham, M. S. and Strømman, A. H. (2016) ‘Nanotechnology for environmentally sustainable electromobility’, _Nature Nanotechnology_. Nature Publishing Group, 11(12), pp. 1039–1051. doi: [10.1038/nnano.2016.237](http://dx.doi.org/10.1038/nnano.2016.237).


Installation
------------

This module should normally be installable via `pip`:

```
pip install git+git://github.com/majeau-bettez/LISET#egg=liset
```
