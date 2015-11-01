/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package machinelearningproject;

import static java.lang.Math.round;
import static java.lang.Math.sqrt;
import java.util.ArrayList;
import java.util.Random;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/**
 *
 * @author Raches
 */
public class RFTree extends Tree{    
    public RFTree()
    {
    
    }
    
    public Instances bootstrap(Instances instances){
        Instances randomInstances = new Instances(instances, instances.numInstances());
        
        for (int i = 0; i < instances.numInstances(); i++){
            int rand = new Random().nextInt(instances.numInstances());
            randomInstances.add(instances.get(rand));
        }
        
        return randomInstances;
    }
        
    @Override
    public Tree buildTree(Instances instances) throws Exception
    {
        Tree tree = new Tree();
        ArrayList<String> availableAttributes = new ArrayList();
        
        //choose random fraction
        int numAttr = instances.numAttributes();
        int k = (int)round(sqrt(numAttr));
        ArrayList<Integer> idxList = new ArrayList();
        for (int i = 0; i < k; i++){
            int rand = new Random().nextInt(numAttr);
            if (idxList.isEmpty() || !idxList.contains(rand)){
                idxList.add(rand);
            } else {
                
            }
        }
        
        int largestInfoGainAttrIdx = -1;
        double largestInfoGainAttrValue = 0.0;
        
        for(int idx = 0; idx < k; idx++){
            if (idx != instances.classIndex()){
                availableAttributes.add(instances.attribute(idx).name());
                System.out.print(instances.attribute(idx).name()+" ; ");
            }
        }
        System.out.println("");
        
        if(instances.numInstances() == 0){
           return null;
        } else if (calculateClassEntropy(instances) == 0.0){
            // all examples have the sama classification
            tree.attributeName = instances.get(0).stringValue(instances.classIndex());
        } else if (availableAttributes.isEmpty()){
            // mode classification
            tree.attributeName = getModeClass(instances, instances.classIndex());
        } else {           
            for (int idx = 0; idx < instances.numAttributes(); idx++){
                if (idx != instances.classIndex()){
                    double attrInfoGain = calculateInformationGain(instances, idx, instances.classIndex());
                    if (largestInfoGainAttrValue < attrInfoGain) {
                        largestInfoGainAttrIdx = idx;
                        largestInfoGainAttrValue = attrInfoGain;
                    }
                }
            }
            
            if (largestInfoGainAttrIdx != -1){
                tree.attributeName = instances.attribute(largestInfoGainAttrIdx).name();
                ArrayList<String> attrValues = new ArrayList();
                for (int i = 0; i < instances.numInstances(); i++){
                    Instance instance = instances.get(i);
                    String attrValue = instance.stringValue(largestInfoGainAttrIdx);
                    if (attrValues.isEmpty() || !attrValues.contains(attrValue)){
                        attrValues.add(attrValue);
                    }
                }

                for (String attrValue: attrValues){
                    Node node = new Node(attrValue);
                    Instances copyInstances = new Instances(instances);
                    copyInstances.setClassIndex(instances.classIndex());
                    int i = 0;
                    while (i < copyInstances.numInstances()){
                        Instance instance = copyInstances.get(i);
                        // reducing examples
                        if (!instance.stringValue(largestInfoGainAttrIdx).equals(attrValue)){
                            copyInstances.delete(i);
                            i--;
                        }
                        i++;
                    }
                    copyInstances.deleteAttributeAt(largestInfoGainAttrIdx);
                    node.subTree = buildTree(copyInstances); 
                    tree.nodes.add(node);
                }
            }
        }
        
        return tree;
    }
    
    public ArrayList<Tree> buildRandomForest(Instances instances, int numTrees) throws Exception
    {
        ArrayList<Tree> dtrees = new ArrayList();
        for (int i = 0; i < numTrees; i++){
            Instances randomInstances = bootstrap(instances);
            dtrees.add(buildTree(instances));
        }
        return dtrees;
    }
    
    public static void main(String[] args) throws Exception{
        RFTree rf = new RFTree();
        DataSource source = new DataSource("D:\\weather-nominal.arff");
        Instances instances = source.getDataSet();
        rf.bootstrap(instances);
    }
}