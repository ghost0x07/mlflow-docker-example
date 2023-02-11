image:
	docker build -t mlflow-docker-example .

run:
	mlflow run . -P n_estimators=20 

.PHONY: image run