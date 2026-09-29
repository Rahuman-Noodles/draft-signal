# Draft Signal shortcuts. `make check` and `make site` are fast and dependency-light;
# `make reproduce` retrains everything (needs torch + matplotlib, takes a few minutes).
.PHONY: check site reproduce serve

check:
	python3 model/check_data.py

site:
	python3 model/build_site_data.py

reproduce:
	cd model && python3 model_pipeline.py && python3 make_charts.py && python3 build_site_data.py

serve:
	python3 -m http.server 8000
