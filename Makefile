install:
	python -m pip install --upgrade pip
	python -m pip install -r app/requirements.txt

train:
	python train.py

eval:
	echo "## Model Metrics" > report.md
	test -f results/results.txt && cat results/results.txt >> report.md || echo "No metrics file found"
	echo "" >> report.md
	echo "## Confusion Matrix Plot" >> report.md
	echo "![Confusion Matrix](./results/model_results.png)" >> report.md

# create an "update" branch, commit model & results, push
update-branch:
	git config --global user.name "github-actions[bot]"
	git config --global user.email "github-actions[bot]@users.noreply.github.com"
	git checkout -B update
	git add model results
	git commit -m "Update model and results [automated]" || echo "No changes to commit"
	git push --force origin update

# deploy: log in to HF CLI & upload files to Space repo
hf-login:
	python -m pip install -U "huggingface_hub[cli]"
	huggingface-cli login --token $(HF_TOKEN)

push-hub:
	cp model/drug_pipeline.joblib drug_pipeline.joblib || echo "Could not copy model to root"
	cp results/results.txt results.txt || echo "Could not copy results to root"
	cp results/model_results.png model_results.png || echo "Could not copy plot to root"
	ls -la *.py *.joblib *.txt *.md requirements.txt || echo "Some files missing"
	huggingface-cli upload shansita-s/Sleep_Quality_Monitoring ./app.py --repo-type=space --commit-message="Upload app.py"
	huggingface-cli upload shansita-s/Sleep_Quality_Monitoring ./drug_pipeline.joblib --repo-type=space --commit-message="Upload model"
	huggingface-cli upload shansita-s/Sleep_Quality_Monitoring ./requirements.txt --repo-type=space --commit-message="Upload requirements"
	huggingface-cli upload shansita-s/Sleep_Quality_Monitoring ./README.md --repo-type=space --commit-message="Upload README"

deploy: hf-login push-hub

# Clean generated files
clean:
	rm -rf model results *.png *.txt report.md
	rm -rf __pycache__ .pytest_cache

# Run the Gradio app locally
run-app:
	cd app && python drug_app.py

# Test the training pipeline
test:
	python -m pytest tests/ -v || echo "No tests directory found"

# Full pipeline: install dependencies, train model, evaluate, and generate report
pipeline: install train eval

.PHONY: install train eval update-branch hf-login push-hub deploy clean run-app test pipeline
