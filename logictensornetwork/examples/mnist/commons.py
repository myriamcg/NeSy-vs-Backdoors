# from collections import defaultdict


# def train(
#         epochs,
#         metrics_dict,
#         ds_train,
#         ds_test,
#         train_step,
#         test_step,
#         csv_path=None,
#         scheduled_parameters=defaultdict(lambda: {})
# ):
#     template = "Epoch {}"
#     for metrics_label in metrics_dict.keys():
#         template += ", %s: {:.4f}" % metrics_label

#     # Open CSV if needed
#     if csv_path is not None:
#         csv_file = open(csv_path, "w+")
#         headers = ",".join(["Epoch"] + list(metrics_dict.keys()))
#         csv_template = ",".join(["{}" for _ in range(len(metrics_dict) + 1)])
#         csv_file.write(headers + "\n")

#     # To store metrics per epoch
#     history = {k: [] for k in metrics_dict.keys()}

#     for epoch in range(epochs):
#         for metrics in metrics_dict.values():
#             metrics.reset_state()

#         for batch_elements in ds_train:
#             train_step(*batch_elements,  **scheduled_parameters[epoch])
#         for batch_elements in ds_test:
#             test_step(*batch_elements, **scheduled_parameters[epoch])

#         metrics_results = [metrics.result() for metrics in metrics_dict.values()]

#         # Store metrics
#         for key, value in zip(metrics_dict.keys(), metrics_results):
#             history[key].append(float(value))

#         print(template.format(epoch, *metrics_results))
#         if csv_path is not None:
#             csv_file.write(csv_template.format(epoch, *metrics_results) + "\n")
#             csv_file.flush()

#     if csv_path is not None:
#         csv_file.close()

#     return history


from collections import defaultdict
import matplotlib.pyplot as plt

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
            train_step(*batch_elements,  **scheduled_parameters[epoch])

        for batch_elements in ds_test:
            test_step(*batch_elements, **scheduled_parameters[epoch])
        all_preds = []
        all_labels = []
        all_preds_x = []
        all_preds_y = []
        all_images_x = []
        all_images_y = []
        for batch in ds_test:
            preds_x, preds_y, preds, labels, images_x, images_y = test_step(*batch)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
            all_preds_x.extend(preds_x.numpy())
            all_preds_y.extend(preds_y.numpy())
            all_images_x.extend(images_x.numpy())
            all_images_y.extend(images_y.numpy())

        # Now print real labels and predictions
        index = 0
        for pred_x, pred_y, real, pred, images_x, images_y in zip(all_preds_x, all_preds_y, all_labels, all_preds, all_images_x, all_images_y):

            print(f" pred_x: {pred_x}, pred_y : {pred_y}, Real: {real}, Predicted: {pred}")
            # plt.subplot(121)
            # plt.imshow(images_x)
            # plt.subplot(122)
            # plt.imshow(images_y)
            # plt.savefig(f"epoch_{epoch}_index_{index}.png")
            index += 1

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
