from collections import defaultdict


def train(
        epochs,
        metrics_dict,
        ds_train,
        ds_test,
        train_step,
        test_step,
        csv_path=None,
        scheduled_parameters=defaultdict(lambda: {})
):
    template = "Epoch {}"
    for metrics_label in metrics_dict.keys():
        template += ", %s: {:.4f}" % metrics_label

    # Open CSV if needed
    if csv_path is not None:
        csv_file = open(csv_path, "w+")
        headers = ",".join(["Epoch"] + list(metrics_dict.keys()))
        csv_template = ",".join(["{}" for _ in range(len(metrics_dict) + 1)])
        csv_file.write(headers + "\n")

    # To store metrics per epoch
    history = {k: [] for k in metrics_dict.keys()}

    for epoch in range(epochs):
        for metrics in metrics_dict.values():
            metrics.reset_state()

        for batch_elements in ds_train:
            train_step(*batch_elements, **scheduled_parameters[epoch])
        for batch_elements in ds_test:
            test_step(*batch_elements, **scheduled_parameters[epoch])

        metrics_results = [metrics.result() for metrics in metrics_dict.values()]

        # Store metrics
        for key, value in zip(metrics_dict.keys(), metrics_results):
            history[key].append(float(value))

        print(template.format(epoch, *metrics_results))
        if csv_path is not None:
            csv_file.write(csv_template.format(epoch, *metrics_results) + "\n")
            csv_file.flush()

    if csv_path is not None:
        csv_file.close()

    return history
