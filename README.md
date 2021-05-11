1. unpack zip archive with datasets ml_100_data.zip
2. change the path to the folder SAROS on yours in

	-saros_convergence.py in the line:
		df = pd.read_csv('/home/sburashnikova/SAROS/datasets/ml_100_data',sep=',',header=None)

	-metrics_saros.py in the lines:

		-df = pd.read_csv('/home/sburashnikova/SAROS/datasets/ml_100_data',sep=',',header=None)

	-run_metrics.sh

3. run saros_convergence.py (results will be in the root folder)
4. run metrics_saros.py (results will be in the folder results -> em)
