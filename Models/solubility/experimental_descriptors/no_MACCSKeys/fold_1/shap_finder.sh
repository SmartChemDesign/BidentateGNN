for env in $(conda env list | grep -v '^#' | awk '{print $1}'); do
    echo "Проверка окружения: $env"
    conda list -n $env | grep "shap"
done