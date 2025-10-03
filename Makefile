install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

train:
	python train.py

eval:
	echo "## Model Metrics" > report.md
	test -f Results/metrics.txt && cat Results/metrics.txt >> report.md || echo "No metrics file found"
	echo "" >> report.md
	echo "## Confusion Matrix Plot" >> report.md
	echo "![Confusion Matrix](./Results/model_results.png)" >> report.md

# create an "update" branch, commit model & results, push
update-branch:
	git config --global user.name "github-actions[bot]"
	git config --global user.email "github-actions[bot]@users.noreply.github.com"
	git checkout -B update
	git add Model Results
	git commit -m "Update model and results [automated]" || echo "No changes to commit"
	git push --force origin update

# deploy: log in to HF CLI & upload files to Space repo
hf-login:
	python -m pip install -U "huggingface_hub[cli]"
	huggingface-cli login --token $(HF_TOKEN) --add-to-git-credential

push-hub:
	cp Model/drug_pipeline.joblib app/drug_pipeline.joblib || echo "Could not copy model to app folder"
hf upload shansita-s/Sleep_Quality_Monitoring ./app --repo-type=space --commit-message="Sync App files"
hf upload shansita-s/Sleep_Quality_Monitoring ./Model --repo-type=space --commit-message="Sync Model"
hf upload shansita-s/Sleep_Quality_Monitoring ./Results --repo-type=space --commit-message="Sync Results"

deploy: hf-login push-hub

# Clean generated files
clean:
	rm -rf Model Results *.png *.txt report.md
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
