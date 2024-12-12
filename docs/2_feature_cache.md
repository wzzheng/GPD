### Feature cache

Preprocess the dataset to accelerate training. The following command generates 1M frames of training data from the whole nuPlan training set. You may need:
- change `cache.cache_path` to suit your condition
- decrease/increase `worker.threads_per_node` depends on your RAM and CPU.

```sh
 export PYTHONPATH=$PYTHONPATH:$(pwd)

 python run_training.py \
    py_func=cache +training=train_planTF \
    scenario_builder=nuplan \
    cache.cache_path=/nuplan/exp/cache_plantf_1M \
    cache.cleanup_cache=true \
    scenario_filter=training_scenarios_1M \
    worker.threads_per_node=40

 #　Process the cache metadata CSV file to generate a quick index.
 python .src/utils/update_cache_metadata_csv.py path/to/your/cache/matedata/xxx_metadata_xxx.csv
```

This process may take some time, be patient (20+hours in my setting).
