import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
import copy


class Node:
    """ Represents a node in our tree
        It has the attribute which is the features based on the split its happening. Also it has the value of the split
        or if it is a leaf then its the label in which it belongs. Finally it has its left and right child (may be None if it is a leaf)
        and the leaf variable which shows if it is a leaf or not.
    """
    def __init__(self, attribute, value, left=None, right=None, leaf=False):
        self.attribute = attribute
        self.value = value
        self.left = left
        self.right = right
        self.leaf = leaf

    def print_node(self):
        """ Print the node in the console
            Args:
            self : only gets the node
        """
        print("attr:", self.attribute, "value:", self.value, "leaf:",
              self.leaf)


def get_entropy(dataset):
    """ Generate entropy of a dataset
    Args:
        dataset (np.array): Dataset with all features and the label being the last column
    Returns:
        entropy(float): the entropy of the given dataset
    """
    labels, counts = np.unique(dataset[:, -1], return_counts=True)
    entropy = 0
    for count in counts:
        label_entropy = (count / len(dataset)) * (np.log2(
            (count / len(dataset))))
        if len(dataset) > 0:
            entropy += label_entropy
        else:
            return 0
    return entropy


def get_remainder(data, split_number):
    """ Generate remainder of a dataset based on a specific split.
    Args:
        data (np.array): Dataset with all features and the label being the last column
        split_number(int): the number by which the whole dataset must be splitted to the left and right subset.
    Returns:
        remainder(float): the remainder of the given dataset, which is weighted entropy of the left and right subsets
    """
    if len(data) == 0:
        return 0
    elif len(data) == 1:
        return get_entropy(data)
    else:
        left_values = data[data[:, 0] < split_number]
        right_values = data[data[:, 0] >= split_number]
        left_entropy = get_entropy(left_values)
        right_entropy = get_entropy(right_values)
        remainder_right = (-1) * (
            (len(right_values) / len(data)) * right_entropy)
        remainder_left = (-1) * ((len(left_values) / len(data)) * left_entropy)
        remainder = remainder_left + remainder_right
        return remainder


def get_best_feature(data):
    """ Finds the best feature and its best split. It finds the best feature and best split among the whole dataset that gives the best remainder
    Args:
        data (np.array): dataset with all features and the label being the last column
    Returns:
        best_feature(int): the best feature number (it is its column number)
        best_split (float): the best number in which the tree should split into left and right subtrees
        best_remainder(float): the best remainder for the whole dataset, its the remainder of the best split of the best feature
    """
    labels = data[:, -1]
    data_2 = data[:, :-1]
    best_remainder = 1000
    best_feature = -1
    best_split = -1
    column_id = 0
    for column in data_2.T:
        combined_array = np.column_stack((column, labels))
        indices = np.argsort(combined_array[:, 0])
        sorted_array = combined_array[indices]
        for index, number in enumerate(sorted_array):
            remainder = get_remainder(sorted_array, number[0])
            if remainder < best_remainder:
                best_remainder = remainder
                best_split = number[0]
                best_feature = column_id
        column_id = column_id + 1
    return (best_feature, best_split, best_remainder)


def decision_tree_learning(dataset, depth):
    """ It trains the decision tree based on the dataset. It is a recursion function, which finds the initial entropy,
        finds the best remainder and if their difference is zero then it creates a leaf. If not, then it breaks the dataset according to the best
        feature and best split of the current dataset, and calls recursively the function decision_tree_learning (itself) on its left and right child
        with the according dataset.
    Args:
        data (np.array): dataset with all features and the label being the last column
        depth (int): the depth of the current node that called the function
    Returns:
        node(Node): the newly created node of the tree
        depth (int): the depth of the whole tree/subtree this node represents
    """
    h_initial = (-1) * get_entropy(dataset)
    best_feature, best_split, remainder = get_best_feature(dataset)
    if ((h_initial - remainder) == 0):
        node = Node(attribute=None, value=dataset[0, -1])
        node.leaf = True
        return (node, depth)
    else:
        left_dataset = dataset[dataset[:, best_feature] < best_split]
        right_dataset = dataset[dataset[:, best_feature] >= best_split]
        node = Node(attribute=best_feature, value=best_split)
        l_branch, l_depth = decision_tree_learning(left_dataset, depth + 1)
        node.left = l_branch
        r_branch, r_depth = decision_tree_learning(right_dataset, depth + 1)
        node.right = r_branch
        return (node, max(l_depth, r_depth))


def find_accuracy(confusion_matrix):
    """ Calculates accuracy of a confusion matrix, by calculating its diagonal
    Args:
        confusion_matrix (np.array): The confusion matrix
    Returns:
        accuracy (float): The accuracy of the confusion matrix
    """
    return (np.trace(confusion_matrix) / np.sum(confusion_matrix))


def find_precision_class(confusion_matrix, class_label):
    """ Calculates precision of a specific class of a confusion matrix.
    Args:
        confusion_matrix (np.array): The confusion matrix
        class_label (int) : The class for which the precision metric must be calculated
    Returns:
        precision (float): The precision of the given class of the confusion matrix
    """
    tp = confusion_matrix[class_label - 1][class_label - 1]
    fp = np.sum(confusion_matrix[:, class_label - 1]) - tp
    return (tp / (tp + fp))


def find_recall_class(confusion_matrix, class_label):
    """ Calculates recall of a specific class of a confusion matrix.
    Args:
        confusion_matrix (np.array): The confusion matrix
        class_label (int) : The class for which the precision metric must be calculated
    Returns:
        recall (float): The recall of the given class of the confusion matrix
    """
    tp = confusion_matrix[class_label - 1][class_label - 1]
    fn = np.sum(confusion_matrix[class_label - 1, :]) - tp
    return (tp / (tp + fn))


def find_f1_score(precision, recall):
    """ Calculates f1-score given the precision and recall
    Args:
        precision (float): The given precision
        recall (float): The given recall
    Returns:
        F1-score (float): The f1-score of the given precision and recall
    """
    return (2 * ((precision * recall) / (precision + recall)))


def predict_one_record(row, node):
    """ Makes a prediction of a given row of the dataset based on a given decision tree. It traverses the tree,
        and based on the data of the given row it goes to the right or the left subtree until it goes to a leaf
        where the predicted label is.
    Args:
        row (np.array): The given row which we want to predict its label
        node (Node): The given decision tree model
    Returns:
        value (int): The label predicted by the model for the specific row
    """
    while (not node.leaf):
        if (row[node.attribute] < (node.value)):
            node = node.left
        else:
            node = node.right
    return node.value


def evaluate(dataset, node):
    """ Makes predictions for the whole dataset based on a given decision tree model. It iterates all the rows of the dataset making predictions
        for each one and then evaluate it and adds the result in the correct position of the confusion matrix. Then it returns the confusion matrix
        as it represents the "whole" prediction and we can calculate all the metrics from it.
    Args:
        dataset (np.array): The given dataset for which we want to predict all its rows' labels
        node (Node): The given decision tree model
    Returns:
        confusion_matrix (np.array): The confusion matrix.
    """
    confusion_matrix = np.zeros((4, 4))
    all_rows = dataset[:, :-1]
    all_labels = dataset[:, -1]
    for index, row in enumerate(all_rows):
        predicted_label = predict_one_record(row, node)
        correct_label = all_labels[index]
        confusion_matrix[int(correct_label) - 1, int(predicted_label) - 1] += 1
    return confusion_matrix


def findAllStatistics(confusion_matrix):
    """ Calculates the accuracy, precision, recall, and f1-score of a given confusion matrix. The precision,recall and f1-score
        are calculated for each label of our dataset(1-4 wifi rooms)
    Args:
        confusion_matrix (np.array): The given confusion matrix
    Returns:
        accuracy (float): The accuracy
        precision (np.array): The precision metric of all the classes
        recall (np.array): The recall metric of all the classes
        f1-score (np.array): The F1-score metric of all the classes
    """
    accuracy = find_accuracy(confusion_matrix)
    precision = np.zeros((4, ))
    recall = np.zeros((4, ))
    f1_score = np.zeros((4, ))
    for i in range(1, 5):
        precision[i - 1] = find_precision_class(confusion_matrix, i)
        recall[i - 1] = find_recall_class(confusion_matrix, i)
        f1_score[i - 1] = find_f1_score(precision[i - 1], recall[i - 1])
    return (accuracy, precision, recall, f1_score)


def prune(root, node, full_dataset_to_eval, current_subset, depth):
    """ Prunes a given tree. It call recursively itself until it goes to a node that its two children are leaf. Calculates
        the validation error of the current tree and the validation error of the tree when it makes that node a leaf. It keeps the one
        with the smallest error. This is done recursively so it checks all the nodes of the tree.
    Args:
        root (Node): The root of the tree so it is easier to evaluate the tree on the dataset
        node (Node): The specific node for which the changes are made/checked
        full_dataset_to_eval (np.array): The whole dataset based on which the tree will be evaluated
        current_subset (np.array): The dataset which the subtree actually contains based on which we traverse the tree/subtree 
            (eg. x0<55 so it will have only the rows with x0<55), also is used to find the new label of the node that its 2 children are leafs.
        depth(int) : The depth of the new pruned tree
    Returns:
        node (Node): The new root of the tree/subtree
        depth(int): The depth of the new pruned tree
    """
    if (node.leaf):
        return node, depth
    depth_l = 0
    depth_r = 0
    if (not node.left.leaf):
        left_dataset = current_subset[
            current_subset[:, node.attribute] < node.value]
        left_branch, depth_l = prune(root, node.left, full_dataset_to_eval,
                                     left_dataset, depth + 1)
        node.left = left_branch
    if (not node.right.leaf):
        right_dataset = current_subset[
            current_subset[:, node.attribute] >= node.value]
        right_branch, depth_r = prune(root, node.right, full_dataset_to_eval,
                                      right_dataset, depth + 1)
        node.right = right_branch

    if node.left.leaf and node.right.leaf:
        error_before = 1 - (find_accuracy(evaluate(full_dataset_to_eval,
                                                   root)))
        old_node = copy.deepcopy(node)
        node.attribute = None
        node.left = None
        node.right = None
        node.leaf = True
        labels, counts = np.unique(current_subset[:, -1], return_counts=True)
        index = np.argmax(counts)
        node.value = labels[index]
        error_after = 1 - (find_accuracy(evaluate(full_dataset_to_eval, root)))
        if (error_after <= error_before):
            return node, depth
        else:
            node = old_node
            return node, depth + 1
    else:
        return node, max(depth_l, depth_r)


def k_fold_split(n_splits, n_instances, random_generator=default_rng()):
    """ Split n_instances into n mutually exclusive splits at random.
    Args:
        n_splits (int): Number of splits
        n_instances (int): Number of instances to split
        random_generator (np.random.Generator): A random generator
    Returns:
        list: a list (length n_splits). Each element in the list should contain a
            numpy array giving the indices of the instances in that split.
    """
    shuffled_indices = random_generator.permutation(n_instances)
    split_indices = np.array_split(shuffled_indices, n_splits)
    return split_indices


def train_test_k_fold(n_folds, n_instances, random_generator=default_rng()):
    """ Generate train and test indices at each fold.
    Args:
        n_folds (int): Number of folds
        n_instances (int): Total number of instances
        random_generator (np.random.Generator): A random generator
    Returns:
        list: a list of length n_folds. Each element in the list is a list (or tuple)
            with two elements: a numpy array containing the train indices, and another
            numpy array containing the test indices.
    """
    split_indices = k_fold_split(n_folds, n_instances, random_generator)
    folds = []
    for k in range(n_folds):
        test_indices = split_indices[k]
        train_indices = np.hstack(split_indices[:k] + split_indices[k + 1:])
        folds.append([train_indices, test_indices])
    return folds


def run_k_fold(dataset):
    """ Runs the k-fold evaluation for the part 3 of the coursework. The k is 10. It splits the dataset to 10 splits
        with one 9 splits it makes/ train the decision tree and evaluates it with the other split. The splits are iterated
        so 10 trees/evaluations will be made and an overall confusion matrix is created which later is divided by the number of
        evaluations that were made (10) and calculate and print the other metrics like confusion matrix, accuracy, precision recall and f1-score for all the
        classes/labels.
    Args:
        dataset (np.array): The dataset which is used in the k-fold.
    Returns:
        Void,as all the evaluation metrics are printed and displayed in console.
    """
    n_fold = 10
    x = dataset[:, :-1]
    y = dataset[:, -1]
    depth_avg = 0
    overall_confusion_matrix = np.zeros((4, 4))
    for i, (train_indices,
            test_indices) in enumerate(train_test_k_fold(n_fold, len(x))):
        x_train = x[train_indices, :]
        y_train = y[train_indices]
        x_test = x[test_indices, :]
        y_test = y[test_indices]
        train_dataset = np.column_stack((x_train, y_train))
        test_dataset = np.column_stack((x_test, y_test))
        model, depth = decision_tree_learning(train_dataset, 1)
        depth_avg += depth
        confusion_matrix = evaluate(test_dataset, model)
        overall_confusion_matrix += confusion_matrix

    depth_avg = depth_avg / n_fold
    overall_confusion_matrix = overall_confusion_matrix / n_fold
    overall_accuracy, overall_precisions, overall_recalls, overall_f1_scores = findAllStatistics(
        overall_confusion_matrix)
    print("Confusion_matrix:", overall_confusion_matrix, "\nAccuracy:",
          overall_accuracy, "\nPrecisions:", overall_precisions, "\nRecalls:",
          overall_recalls, "\nF1-scores:", overall_f1_scores, "\nDepth:",
          depth_avg)


def run_k_fold_with_pruning(dataset):
    """ Runs the k-fold evaluation for the part 4 of the coursework, so it adds the pruning process in it. The k is 10. 
        It splits the dataset to 10 splits with 1 being the test set. Then iterates the other 9 and gets 1 for validate set and 8 for training.
        The tree is created with the training data, then pruned with the validate and tested with the testing data. The initial test split is also iterated.
        In the end 90 (10*9) trees are created, the average confusion matrix is computed and the different metrics are computed from that.
        Finally, it prints the metrics of confusion matrix, accuracy,precision recall and f1-score for all the
        classes/labels.
    Args:
        dataset (np.array): The dataset which is used in the k-fold.
    Returns:
        Void, as all the evaluation metrics are printed and displayed in console.
    """
    n_fold = 10
    split_indices = k_fold_split(n_fold, len(dataset))
    overall_confusion_matrix = np.zeros((4, 4))
    depth_avg = 0
    old_depth_avg = 0
    for initial in range(n_fold):
        x = dataset[:, :-1]
        y = dataset[:, -1]
        test_indices = split_indices[initial]
        x_test = x[test_indices, :]
        y_test = y[test_indices]
        test_dataset = np.column_stack((x_test, y_test))
        folds = []
        for k in range(n_fold):
            if k != initial:
                val_indices = split_indices[k]
                train_indices = np.hstack(split_indices[:k] +
                                          split_indices[k + 1:])
                folds.append([train_indices, val_indices])
        for i, (train_indices, val_indices) in enumerate(folds):
            # Get the dataset from the correct splits
            x_train = x[train_indices, :]
            y_train = y[train_indices]
            x_val = x[val_indices, :]
            y_val = y[val_indices]
            train_dataset = np.column_stack((x_train, y_train))
            val_dataset = np.column_stack((x_val, y_val))
            model, old_depth = decision_tree_learning(train_dataset, 1)
            root = model
            pruned_model, depth = prune(root, model, val_dataset,
                                        train_dataset, 1)
            old_depth_avg += old_depth
            depth_avg += depth
            confusion_matrix = evaluate(test_dataset, pruned_model)
            overall_confusion_matrix += confusion_matrix

    old_depth_avg = old_depth_avg / (n_fold * (n_fold - 1))
    depth_avg = depth_avg / (n_fold * (n_fold - 1))
    overall_confusion_matrix = overall_confusion_matrix / (n_fold *
                                                           (n_fold - 1))
    overall_accuracy, overall_precisions, overall_recalls, overall_f1_scores = findAllStatistics(
        overall_confusion_matrix)
    print("Confusion_matrix:", overall_confusion_matrix, "\nAccuracy:",
          overall_accuracy, "\nPrecisions:", overall_precisions, "\nRecalls:",
          overall_recalls, "\nF1-scores:", overall_f1_scores, "\nDepth:",
          depth_avg)


def visualize_tree(node, depth=0, parent_x=0, parent_y=0, position=0):
    """ Makes and prints a visualization of a tree.
    Args:
        node (Node): The root of the tree that will be visualized
        depth (np.array):  The depth of the node
        parent_x(int): The x position of the father of the node
        parent_y(int): The y position of the father of the node
        position(str): Defines the position which is adjusted based on the depth and the previous position
    """
    if node is not None:
        # Determine the coordinates for this node
        x = position
        y = -depth  # adjust for spacing
        # Plot decision nodes
        if not node.leaf:
            # Plot the decision split (feature and threshold)
            plt.text(x,
                     y,
                     f'Feature {node.attribute}\n<= {node.value:.2f}',
                     fontsize=4.5,
                     ha='center',
                     va='center',
                     bbox=dict(boxstyle='round,pad=0.4', edgecolor='black'))

            # Recursively plot the left child with position adjusted to the left
            visualize_tree(node.left,
                           depth + 1,
                           x,
                           y,
                           position=(position - 1 / (depth + 1)))
            # Draw a line connecting to the left child
            plt.plot([x, position - 1 / (depth + 1)], [y, y - 1],
                     color='black')
            # Recursively plot the right child with position adjusted to the right
            visualize_tree(node.right,
                           depth + 1,
                           x,
                           y,
                           position=(position + 1 / (depth + 1)))

            # Draw a line connecting to the right child
            plt.plot([x, position + 1 / (depth + 1)], [y, y - 1],
                     color='black')
        # Plot leaf nodes
        else:
            # Plot the predicted class label
            plt.text(x,
                     y,
                     f'Class {node.value}',
                     ha='center',
                     va='center',
                     fontsize=4.5,
                     bbox=dict(facecolor='yellow',
                               boxstyle='round,pad=0.4',
                               edgecolor='black'))


def execute_non_pruning_evaluation():
    """ Main function for part 2 and 3 of the coursework. Loads the clean dataset, runs k-fold validation(without pruning) for it, and then does the same for
        the noisy dataset. According messages are printed in the console.
    Args:
        Void.
    Returns:
        Void, as all the evaluation metrics are printed and displayed in console.
    """
    print("Part 3: Non pruning evaluation")
    print("Clean dataset")
    dataset = np.loadtxt(fname="clean_dataset.txt")
    run_k_fold(dataset)
    print("\nNoisy dataset")
    noisy_dataset = np.loadtxt(fname="noisy_dataset.txt")
    run_k_fold(noisy_dataset)


def execute_pruning_evaluation():
    """ Main function for part 4 of the coursework. Loads the clean dataset, runs k-fold validation(with pruning) for it, and then does the same for
        the noisy dataset. According messages are printed in the console.
    Args:
        Void.
    Returns:
        Void, as all the evaluation metrics are printed and displayed in console.
    """
    print("Part 4: Pruning evaluation")
    print("Clean dataset")
    dataset = np.loadtxt(fname="clean_dataset.txt")
    run_k_fold_with_pruning(dataset)

    print("\nNoisy dataset")
    noisy_dataset = np.loadtxt(fname="noisy_dataset.txt")
    run_k_fold_with_pruning(noisy_dataset)


def show_trained_tree():
    print("Bonus part visualization of tree trained on clean dataset")
    clean_dataset = np.loadtxt(fname="clean_dataset.txt")
    tree, _ = decision_tree_learning(clean_dataset, depth=1)
    plt.figure(figsize=(100, 70))
    visualize_tree(tree)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    """ Main function for the whole coursework. Just calls function execute_non_pruning_evaluation for steps 2 and 3 of the coursework.
        Then it call the execute_pruning_evaluation to run the pruning part - part 4 of the coursework and in the end it calls show_trained_tree to show the visualization
        for the bonus part.
    """
    execute_non_pruning_evaluation()
    execute_pruning_evaluation()
    show_trained_tree()
