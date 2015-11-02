/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package machinelearningproject;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author Raches
 */
public class Tree implements Serializable{
    // reference source code: http://afewguyscoding.com/2010/03/id3-decision-trees-java/
    class Node implements Serializable{
        public String value;
        public Tree subTree;
              
        public Node()
        {
            this.subTree = new Tree();
            this.value = "";
        }
        public Node(String value)
        {
            this.subTree = new Tree();
            this.value = value;
        }
    }
    
    public String attributeName;
    public ArrayList<Node> nodes;
        
    public Tree(){
        this.nodes = new ArrayList();
        this.attributeName = "";
    }
    
    public int getIndex(String attributeName)
    {
        int indexFound = -1;
        int i = 0;
        
        while(i < nodes.size() && indexFound == -1)
        {
            if(nodes.get(i).subTree.attributeName.equals(attributeName))
            {
                indexFound = i;
            }
            i++;
        }
        return indexFound;
    }
    
    public boolean isLeaf() 
    {
        return (nodes.isEmpty());
    }
    
    public String getModeClass(Instances instances, int classIdx){
        HashMap<String, Integer> classMap = new HashMap<>();
        int numInstances = instances.size();
        
        for (int i = 0; i < numInstances; i++){
            Instance instance = instances.get(i);
            String key = instance.stringValue(classIdx);
            if (classMap.isEmpty() || !classMap.containsKey(key)) {
                classMap.put(key, 1);
            } else {
                if (classMap.containsKey(key)){
                    classMap.put(key, classMap.get(key) + 1);
                }
            }
        }
        Iterator<String> keySetIterator = classMap.keySet().iterator();
        String modeClass = "";
        int count = 0;
        while(keySetIterator.hasNext()){ 
            String key = keySetIterator.next();
            System.out.println("key: " + key + " value: " + classMap.get(key));
            if (count < classMap.get(key)){
                modeClass = key;
                count = classMap.get(key);
            }
        }
        return modeClass;
    }
    
    public double calculateEntropy(Instances instances, int attrIdx)
    {
        HashMap<String, Integer> classMap = new HashMap<>();
        double entropy = (double)0;
        int numInstances = instances.size();
        
        for (int i = 0; i < numInstances; i++){
            Instance instance = instances.get(i);
            String key = instance.stringValue(attrIdx);
            if (classMap.isEmpty() || !classMap.containsKey(key)) {
                classMap.put(key, 1);
            } else {
                if (classMap.containsKey(key)){
                    classMap.put(key, classMap.get(key) + 1);
                }
            }
        }
        
        Iterator<String> keySetIterator = classMap.keySet().iterator(); 
        while(keySetIterator.hasNext()){ 
            String key = keySetIterator.next();
            // reference source code http://onoffswitch.net/building-decision-tree/
            double prob = (double)classMap.get(key) / (double)numInstances;
            entropy -= prob * (Math.log(prob)/Math.log(2));
        }
        
        return entropy;
    }
    
    public double calculateClassEntropy(Instances instances){
        return calculateEntropy(instances, instances.numAttributes()-1);
    }
    
    public double calculateInformationGain(Instances instances, int attrIdx, int classIdx) throws Exception       
    {
        HashMap<String, Integer> attrCount = new HashMap<>();
        HashMap<String, Integer> attrClassCount = new HashMap<>();
        int numInstances = instances.size();
        
        for (int i = 0; i < numInstances; i++){
            Instance instance = instances.get(i);
            
            String attrKey = instance.stringValue(attrIdx);
            if (attrCount.isEmpty() || !attrCount.containsKey(attrKey)) {
                attrCount.put(attrKey, 1);
            } else {
                if (attrCount.containsKey(attrKey)){
                    attrCount.put(attrKey, attrCount.get(attrKey) + 1);
                }
            }
            
            String attrClassKey = instance.stringValue(attrIdx)+"-"+instance.stringValue(classIdx);
            if (attrClassCount.isEmpty() || !attrClassCount.containsKey(attrClassKey)) {
                attrClassCount.put(attrClassKey, 1);
            } else {
                if (attrClassCount.containsKey(attrClassKey)){
                    attrClassCount.put(attrClassKey, attrClassCount.get(attrClassKey) + 1);
                }
            }
        }
        double attrEntropy = (double)0;
        
        Iterator<String> attrKeySetIterator = attrCount.keySet().iterator();
        while(attrKeySetIterator.hasNext()){
            String attrKey = attrKeySetIterator.next();
            double bufferEntropy = (double)0;
            Iterator<String> keySetIterator = attrClassCount.keySet().iterator();
            while(keySetIterator.hasNext()){ 
                String key = keySetIterator.next();
                String[] keys = key.split("-");
                String attrValue = keys[0];
                if (attrKey.equals(attrValue)) {
                    double prob = (double)attrClassCount.get(key) / (double)attrCount.get(attrKey);
                    bufferEntropy -= prob * (Math.log(prob)/Math.log(2));
                }
            }
            attrEntropy += (attrCount.get(attrKey)/(double)numInstances)*bufferEntropy;
        }
        double classEntropy = calculateEntropy(instances, classIdx);
        
        return (classEntropy-attrEntropy);
    }
    
    public Tree buildTree(Instances instances) throws Exception
    {
        Tree tree = new Tree();
        ArrayList<String> availableAttributes = new ArrayList();
        
        int largestInfoGainAttrIdx = -1;
        double largestInfoGainAttrValue = 0.0;
        
        for(int idx = 0; idx < instances.numAttributes(); idx++){
            if (idx != instances.classIndex()){
                availableAttributes.add(instances.attribute(idx).name());
            }
        }
        
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
    
    public String traverseTree(Instance instance)
    {
        String attrValue= "";
        Tree buffTree = this;
        while(!buffTree.isLeaf()){     
            //get attribute value of an instance
            for(int i = 0; i < instance.numAttributes(); i++) {
                if(instance.attribute(i).name().equals(buffTree.attributeName)){
                    attrValue = instance.stringValue(i);
                    break;
                }
            }

            //compare attribute with node value
            for(int i = 0; i < buffTree.nodes.size(); i++) {
                if(attrValue.equals(buffTree.nodes.get(i).value)){
                    buffTree = buffTree.nodes.get(i).subTree;
                    break;
                }
            }                
        }
        
        //isLeaf
        attrValue = buffTree.attributeName;
        
        return attrValue;
    }
    
    public void printTree(Tree tree)
    {
        printTree(tree, 0);
    }
    
    public void printTree(Tree tree, int depth)
    {
        System.out.print("("+depth+")");
        for(int i = 0; i <= depth; i++){
            System.out.print("- - ");
        }
        System.out.println(tree.attributeName);
        
        if (!tree.isLeaf()){
            for(int i = 0; i < tree.nodes.size(); i++){
                System.out.print("("+depth+")");
                for(int j = 0; j <= depth; j++){
                    System.out.print(" | | ");    
                }
                System.out.println(tree.nodes.get(i).value);
                printTree(tree.nodes.get(i).subTree, depth+1);
            }
        }
    }
}
