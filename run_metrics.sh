#echo 'installing tensorflow'
#export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.1-cp34-cp34m-linux_x86_64.whl
#pip3 install --upgrade --user $TF_BINARY_URL

    cd java/src
    echo 'making relevance vector'
    javac -cp ../binaries/commons-lang3-3.5.jar  preProcess/ConvertIntoRelVecGeneralized.java preProcess/InputOutput.java
    java -cp . preProcess.ConvertIntoRelVecGeneralized /home/sburashnikova/SAROS/results/gt /home/sburashnikova/SAROS/results/pr /home/sburashnikova/SAROS/results/rv/relevanceVector_ml_1m 10
    cd -
    echo 'compute offline metrics'
    python3 compOfflineEvalMetrics.py /home/sburashnikova/SAROS/results/ ml_1m

