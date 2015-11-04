/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */ 

package machinelearningproject;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;

/**
 *
 * @author Raches
 */
public class MachineLearningProject {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        // TODO code application logic here
        DataSource source = new DataSource("D:\\spambase.arff");
//        DataSource source = new DataSource("D:\\weather-nominal.arff");
        Instances instances = source.getDataSet();
        int numAttr = instances.numAttributes();
        instances.setClassIndex(instances.numAttributes()-1);
        
        int runs = 5;
        int seed = 15;
        for (int i = 0; i < runs; i++) {
            //randomize data
            seed = seed+1;                       // the seed for randomizing the data
            Random rand = new Random(seed);       // create seeded number generator
            Instances randData = new Instances(instances);   // create copy of original data
            Collections.shuffle(randData);

            Evaluation evalDTree = new Evaluation(randData);
            Evaluation evalRF = new Evaluation(randData);
            Evaluation evalSVM = new Evaluation(randData);

            int folds = 10;
            for (int n = 0; n < folds; n++) {
                Instances train = randData.trainCV(folds, n, rand);
                Instances test = randData.testCV(folds, n);
                //instantiate classifiers
                DecisionTree dtree = new DecisionTree();
                RandomForest rf = new RandomForest(100);
                SMO svm = new SMO();
                RBFKernel rbfKernel = new RBFKernel();
                double gamma = 0.70;
                rbfKernel.setGamma(gamma);
                
                dtree.buildClassifier(train);
                rf.buildClassifier(train);
                svm.buildClassifier(train);
                
                evalDTree.evaluateModel(dtree, test);
                evalRF.evaluateModel(rf, test);
                evalSVM.evaluateModel(svm, test);
            }
            System.out.println("=== Decision Tree Evaluation ===");
            System.out.println(evalDTree.toSummaryString());
            System.out.println(evalDTree.toClassDetailsString());
            System.out.println(evalDTree.toMatrixString());
            
            System.out.println("=== Random Forest Evaluation ===");
            System.out.println(evalRF.toSummaryString());
            System.out.println(evalRF.toClassDetailsString());
            System.out.println(evalRF.toMatrixString());
            
            System.out.println("=== SVM Evaluation ===");
            System.out.println(evalSVM.toSummaryString());
            System.out.println(evalSVM.toClassDetailsString());
            System.out.println(evalSVM.toMatrixString());
        }
    }
    
}
