# Datasets: access and terms (checked 2026-10-11)

Only what the official pages state is recorded. "Not stated" means the page I could read says nothing; it does NOT mean unrestricted. Re-read each page yourself before publishing and cite every dataset.

| Dataset | Role | Official page | Terms found on page | Cite |
|---|---|---|---|---|
| FIVES (800 images, multi-disease) | train | figshare 19688169 | No license text visible on the page I could read; asks users to cite. Check the figshare license field yourself. | Jin et al., Sci Data 9, 475 (2022), doi:10.1038/s41597-022-01564-3 |
| DRIVE (40 images) | test | drive.grand-challenge.org | Landing page states no license/citation terms; test-set labels are withheld by the challenge site (predictions are submitted there). | Staal et al., IEEE TMI 2004 |
| STARE (20 labelled images) | test | cecas.clemson.edu/~ahoover/stare | No license or usage restriction stated; no citation demanded; contact author for data/code requests. | Hoover et al., IEEE TMI 2000 |
| HRF (45 images) | test | www5.cs.fau.de/research/data/fundus-images | "can be used freely for research purposes"; Creative Commons 4.0 Attribution; no login. | Budai et al., Int J Biomed Imaging 2013 |
| CHASE_DB1 (28 images) | test | blogs.kingston.ac.uk/retinal/chasedb1 | NOT CHECKED: the site blocked automated reading. Open it manually. | Owen et al., IOVS 2009 |

## Open items
- DRIVE: the withheld test labels matter. For cross-dataset *testing* we need labels, so use the DRIVE *training* set (20 labelled images) as an evaluation set, or the labelled test images only if you obtained them under the site's terms.
- Do not copy the DRIVE/STARE images bundled inside the VGA-Net GitHub repo into this project: that repo has no license file. Download from the official sources.
- Citation years above are from memory of the standard references; verify against the official pages.
