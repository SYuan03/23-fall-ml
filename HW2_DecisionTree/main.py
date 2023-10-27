import dataLoad
import decisionTree
import sys
from sklearn.metrics import f1_score
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    stdout = sys.stdout
    sys.stdout = open("log.txt", "w")
    tree = decisionTree.Tree()
    print("Info : Load train data")
    data, target = dataLoad.train_file_read("./dataResource/training.csv")
    print("Info : Load valid data")
    valid_data, valid_target = dataLoad.train_file_read("./dataResource/validation.csv")
    print("Info : Load test data")
    test_data = dataLoad.test_file_read("./dataResource/testing.csv")
    print("\n\n\n\n\nInfo : begin to train")
    tree.train(data, target)
    valid_output = tree.valid(valid_data)
    print("Info : decision tree output")
    print(valid_output)
    print("Info : valid set expect")
    print(valid_target)
    print("\n\n\n\n\n\nInfo : Load test result")
    test_output = tree.valid(test_data)
    dataLoad.write_test_result("./dataResource/testing.csv", test_output)
    tree.toString()
    print("Info : MicroF1 score " + str(f1_score(valid_target, valid_output, average="micro")))
    print("Info : MacroF1 score " + str(f1_score(valid_target, valid_output, average="macro")))
    sys.stdout = stdout
