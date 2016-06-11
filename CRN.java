package weka.classifiers.jx_w;

import java.io.FileReader;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Random;

import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.ProtectedProperties;

public class CRN_test extends AbstractClassifier {

	/**
	 * The dataset header for the purposes of printing out a semi-intelligible model
	 */
	protected Instances m_Instances;

	/** The number of classes (or 1 for numeric class) */
	protected int m_NumClasses;

	protected ArrayList<ArrayList<Attribute>> S;

	@Override
	public void buildClassifier(Instances data) throws Exception {

		System.out.println("训练集 : " + data);

		// remove instances with missing class
		data = new Instances(data);
		data.deleteWithMissingClass();

		m_NumClasses = data.numClasses();

		// Copy the instances
		m_Instances = new Instances(data);

		S = learning_feature_sets(data);
		System.out.println("S: " + S);
	}

	@Override
	public double classifyInstance(Instance instance) throws Exception {
		System.out.println("待分类实例: " + instance);
		double minIP = Double.POSITIVE_INFINITY;
		double maxPriority = 0.0;
		HashMap<ArrayList<Attribute>, Double> VandIP = new HashMap<ArrayList<Attribute>, Double>();
		HashMap<ArrayList<Attribute>, Double> VandPriority = new HashMap<ArrayList<Attribute>, Double>();
		HashMap<Object, Integer> classesAndInt = getclassesAndInt();
		HashSet<Object> CLS = new HashSet<Object>();

		for (ArrayList<Attribute> V : S) {
			Enumeration<Object> enu = m_Instances.classAttribute().enumerateValues();
			int[] Neighb = new int[m_NumClasses];
			int i = 0;
			while (enu.hasMoreElements()) {
				Object tempObject = enu.nextElement();
				ArrayList<Instance> Dc = getDc(m_Instances, tempObject);
				Neighb[i] = Neighbors(instance, V, Dc);
				i++;
			}

			if (Impurity(Neighb) < minIP)
				minIP = Impurity(Neighb);
			VandIP.put(V, Impurity(Neighb));
		}

		if (minIP == Double.POSITIVE_INFINITY) {
			int maxInstancesCount = 0;
			Object CLV = new Object();
			Enumeration<Object> enu = m_Instances.classAttribute().enumerateValues();
			while (enu.hasMoreElements()) {
				Object tempObject = enu.nextElement();
				ArrayList<Instance> Dc = getDc(m_Instances, tempObject);
				if (Dc.size() > maxInstancesCount) {
					maxInstancesCount = Dc.size();
					CLV = tempObject;
				}
			}
			return (double) classesAndInt.get(CLV);
		} else {
			for (ArrayList<Attribute> V : VandIP.keySet()) {
				if (VandIP.get(V) == minIP) {
					double priority = getNCLV(instance, V) / V.size();

					VandPriority.put(V, priority);
					if (priority > maxPriority)
						maxPriority = priority;
				} else {
					VandPriority.put(V, 0.0);
				}
			}

		}
		for (ArrayList<Attribute> V : VandPriority.keySet()) {
			if (VandPriority.get(V) == maxPriority) {
				CLS.add(getCLV(instance, V));
			}
		}
		if (CLS.size() == 1) {
			Iterator<Object> iterator = CLS.iterator();
			while (iterator.hasNext()) {
				return (double) classesAndInt.get(iterator.next());
			}
		} else {
			int minInstances = Integer.MAX_VALUE;
			Object objectOfminInstances = new Object();
			for (Object obj : CLS) {
				if (getDc(m_Instances, obj).size() < minInstances) {
					minInstances = getDc(m_Instances, obj).size();
					objectOfminInstances = obj;
				}
			}
			return (double) classesAndInt.get(objectOfminInstances);
		}
		return -1;
	}

	// 返回一个类别与整数之间对应的HashMap
	private HashMap<Object, Integer> getclassesAndInt() {
		HashMap<Object, Integer> classesAndInt = new HashMap<Object, Integer>();
		Enumeration<Object> enu = m_Instances.classAttribute().enumerateValues();
		int i = 0;
		while (enu.hasMoreElements()) {
			classesAndInt.put(enu.nextElement(), i);
			i++;
		}
		return classesAndInt;
	}

	// 用于计算Nc(V)
	private int Neighbors(Instance I, ArrayList<Attribute> V, ArrayList<Instance> Dc) {
		int neighbors = 0;
		for (Instance instance : Dc) {
			if (isNeighbors(I, instance, V)) {
				neighbors++;
			}
		}
		return neighbors;
	}

	// 用于计算IP(V)
	public static double Impurity(int[] Neighb) {
		double sum = 0;
		double IP = 0;
		for (int i = 0; i < Neighb.length; i++) {
			sum += Neighb[i];
		}
		if (sum == 0)
			return Double.POSITIVE_INFINITY;
		for (int i = 0; i < Neighb.length; i++) {
			if (Neighb[i] == 0)
				IP += 0;
			else {
				IP += (Neighb[i] / sum) * (Math.log(Neighb[i] / sum) / Math.log(2));
			}
		}
		return -IP;
	}

	// 用于求得Nc(V)中最大者的类标签，即候选标签
	private Object getCLV(Instance I, ArrayList<Attribute> V) {
		Enumeration<Object> enu = I.classAttribute().enumerateValues();
		int maxNeighb = 0;
		Object CLV = new Object();
		while (enu.hasMoreElements()) {
			Object tempObject = enu.nextElement();
			ArrayList<Instance> Dc = getDc(m_Instances, tempObject);
			if (Neighbors(I, V, Dc) > maxNeighb) {
				maxNeighb = Neighbors(I, V, Dc);
				CLV = tempObject;
			}
		}
		return CLV;
	}

	// 用于求得实例I在V上的候选标签CLV的邻居数量，即NCLV
	private double getNCLV(Instance I, ArrayList<Attribute> V) {
		Enumeration<Object> enu = I.classAttribute().enumerateValues();
		int maxNeighb = 0;
		while (enu.hasMoreElements()) {
			Object tempObject = enu.nextElement();
			ArrayList<Instance> Dc = getDc(m_Instances, tempObject);
			if (Neighbors(I, V, Dc) > maxNeighb) {
				maxNeighb = Neighbors(I, V, Dc);
			}
		}
		return (double) maxNeighb;
	}

	/**
	 * @param data
	 *            训练集数据Instances data 找到关于实例集的特征属性集合，对应算法训练阶段的整个过程
	 */
	public ArrayList<ArrayList<Attribute>> learning_feature_sets(Instances data) {
		ArrayList<ArrayList<Attribute>> S = new ArrayList<ArrayList<Attribute>>();
		Enumeration<Object> enu = data.classAttribute().enumerateValues();
		while (enu.hasMoreElements()) {
			Object tempElement = enu.nextElement();
			ArrayList<Instance> temp = getDc(data, tempElement);
			while (!temp.isEmpty()) {
				ArrayList<Instance> DcN = getDcN(data, tempElement);
				int count = temp.size();
				Instance instance = temp.get(new Random().nextInt(count));
				System.out.println("从Temp中选出的实例instance = " + instance);
				ArrayList<Attribute> tempV = learning_one_feature_set(data, instance, DcN);
				if (!containsV(S, tempV)) {
					S.add(tempV);
				}
				updataTemp(temp, instance, tempV);
			}
		}
		return S;
	}

	public ArrayList<Attribute> learning_one_feature_set(Instances data, Instance I, ArrayList<Instance> DcN) {
		ArrayList<Attribute> V = new ArrayList<Attribute>();
		ArrayList<Attribute> attribute_set = new ArrayList<Attribute>();
		Enumeration<Attribute> enu = data.enumerateAttributes();
		while (enu.hasMoreElements()) {
			attribute_set.add(enu.nextElement());
		}
		while (!DcN.isEmpty() && !attribute_set.isEmpty()) {
			Attribute min_attribute = getMinQ(data, I, attribute_set, DcN);
			boolean isUpdateDcN = updateDcN(min_attribute, I, DcN);
			if (!V.contains(min_attribute) && isUpdateDcN)
				V.add(min_attribute);
			attribute_set.remove(min_attribute);
		}
		return V;
	}

	// 用于更新DcN，去掉DcN中实例I根据属性A可区分的实例
	private boolean updateDcN(Attribute min_attribute, Instance I, ArrayList<Instance> DcN) {
		ArrayList<Instance> temp = new ArrayList<Instance>();
		for (Instance instance : DcN) {
			if (!instance.stringValue(min_attribute).equals(I.stringValue(min_attribute))) {
				temp.add(instance);
			}
		}
		for (Instance instance : temp) {
			DcN.remove(instance);
		}
		if (temp.isEmpty())
			return false;
		return true;
	}

	// 用于判定属性集合V是否在属性集合S中存在
	private boolean containsV(ArrayList<ArrayList<Attribute>> S, ArrayList<Attribute> tempV) {
		for (ArrayList<Attribute> temp : S) {
			if (temp.size() == tempV.size()) {
				int i = 0;
				for (Attribute att : temp) {
					if (tempV.contains(att))
						i++;
				}
				if (i == temp.size())
					return true;
			}
		}
		return false;
	}

	// 用于去除实例I在特征集合V上的邻居
	private void updataTemp(ArrayList<Instance> temp, Instance I, ArrayList<Attribute> tempV) {
		ArrayList<Instance> removedInstances = new ArrayList<Instance>();
		for (Instance instance : temp) {
			if (isNeighbors(I, instance, tempV)) {
				removedInstances.add(instance);
			}
		}
		for (Instance instance : removedInstances) {
			temp.remove(instance);
		}
	}

	// 判定实例I与实例IOfTemp在属性集合V上时候是邻居，他们都属于同一个类别
	private boolean isNeighbors(Instance I, Instance IOfTemp, ArrayList<Attribute> V) {
		for (Attribute att : V) {
			if (!I.stringValue(att).equals(IOfTemp.stringValue(att)))
				return false;
		}
		return true;
	}

	// 用于求得实例I在所有A中的Q(A,DcN,D)的最小值,返回最新Q对应的A
	private Attribute getMinQ(Instances data, Instance I, ArrayList<Attribute> attribute_set, ArrayList<Instance> DcN) {
		double minQ = Double.POSITIVE_INFINITY;
		int minIndex = 0;
		for (int i = 0; i < attribute_set.size(); i++) {
			if (quality(data, attribute_set.get(i), I, DcN) < minQ) {
				minQ = quality(data, attribute_set.get(i), I, DcN);
				minIndex = i;
			}
		}

		return attribute_set.get(minIndex);
	}

	// 用于求得Q(A,DcN,D)
	private double quality(Instances data, Attribute att, Instance I, ArrayList<Instance> DcN) {
		double count = Count(data, att, I, DcN);
		if (count == 0)
			return Double.POSITIVE_INFINITY;
		else {
			return ESI(data, att) / count;
		}
	}

	// 用于求得Dc
	private ArrayList<Instance> getDc(Instances data, Object classValue) {
		ArrayList<Instance> Dc = new ArrayList<Instance>();
		Enumeration<Instance> enu = data.enumerateInstances();
		while (enu.hasMoreElements()) {
			Instance instance = enu.nextElement();
			if (instance.stringValue(data.classAttribute()).equals(classValue)) {
				Dc.add(instance);
			}
		}
		return Dc;
	}

	// 用于求得DcN
	private ArrayList<Instance> getDcN(Instances data, Object classValue) {
		ArrayList<Instance> DcN = new ArrayList<Instance>();
		Enumeration<Instance> enu = data.enumerateInstances();
		while (enu.hasMoreElements()) {
			Instance instance = enu.nextElement();
			if (!instance.stringValue(data.classAttribute()).equals(classValue)) {
				DcN.add(instance);
			}
		}
		return DcN;
	}

	/**
	 * @param data
	 *            训练集数据Instances data
	 * @param att
	 *            属性Attribute att 用于计算实例集的Entropy和Split_Info值
	 */
	public double ESI(Instances data, Attribute att) {
		int totalInstances = data.numInstances();
		Enumeration<Object> enu = att.enumerateValues();
		double entropy = 0;
		double split_info = 0;
		while (enu.hasMoreElements()) {
			Object obj = enu.nextElement();
			Enumeration<Object> enu1 = data.classAttribute().enumerateValues();
			double sumOfClasses = 0;
			while (enu1.hasMoreElements()) {
				Object cls = enu1.nextElement();
				double temp = countClassofObj(data, att, obj, cls) / countAllofObj(data, att, obj);
				double temp1;
				if (temp == 0) {
					temp1 = 0;
				} else
					temp1 = temp * (Math.log(temp) / Math.log(2));
				sumOfClasses += temp1;
			}
			entropy += (countAllofObj(data, att, obj) / totalInstances) * sumOfClasses;
			double temp = Math.log(countAllofObj(data, att, obj) / totalInstances) / Math.log(2);
			split_info += temp * (countAllofObj(data, att, obj) / totalInstances);
		}
		return entropy * split_info;
	}

	// 计算D*i
	private double countAllofObj(Instances data, Attribute att, Object obj) {
		int sum = 0;
		if (obj instanceof String) {
			Enumeration<Instance> enu = data.enumerateInstances();
			while (enu.hasMoreElements()) {
				if (enu.nextElement().stringValue(att).equals(obj))
					sum++;
			}
		}
		return sum;
	}

	// 计算Dci
	private double countClassofObj(Instances data, Attribute att, Object obj, Object cls) {
		int sum = 0;
		if (obj instanceof String) {
			Enumeration<Instance> enu = data.enumerateInstances();
			while (enu.hasMoreElements()) {
				Instance instance = enu.nextElement();
				if (instance.stringValue(instance.classAttribute()).equals(cls)) {
					if (instance.stringValue(att).equals(obj))
						sum++;
				}
			}
		}
		return sum;
	}

	// 用于计算C(A,DcN)
	private int Count(Instances data, Attribute att, Instance I, ArrayList<Instance> DcN) {
		int count = 0;
		for (Instance instance : DcN) {
			if (!instance.stringValue(att).equals(I.stringValue(att)))
				count++;
		}
		return count;
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {

		FileReader fr = null;
		Instances data = null;
		try {
			fr = new FileReader("D:\\Program Files\\Weka-3-8\\data\\weather.nominal.arff");
		} catch (Exception e) {
			e.printStackTrace();
		}
		try {
			data = new Instances(fr);
			data.setClassIndex(data.numAttributes() - 1);
		} catch (Exception e) {
			e.printStackTrace();
		}
		CRN_test crn = new CRN_test();

		// crn.ESI(data,data.attribute(0));

		Instances data_copy = new Instances(data);
		Instance instance = data_copy.remove(5);
		crn.buildClassifier(data_copy);

		System.out.println("instance " + instance);
		double d = crn.classifyInstance(instance);
		if (d == 0)
			System.out.println("分类结果: yes");
		else if (d == 1)
			System.out.println("分类结果: no");
		else
			System.out.println("分类结果: 未能正确分类");

		System.out.println("over");
	}
}
