package Praktika3;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import static weka.classifiers.lazy.IBk.TAGS_WEIGHTING;
import static weka.classifiers.lazy.IBk.WEIGHT_NONE;
import static weka.classifiers.lazy.IBk.WEIGHT_INVERSE;
import static weka.classifiers.lazy.IBk.WEIGHT_SIMILARITY;

import weka.core.EuclideanDistance;
import weka.core.Instances;
import weka.core.ManhattanDistance;
import weka.core.MinkowskiDistance;
import weka.core.SelectedTag;
import weka.core.SerializationHelper;
import weka.core.Tag;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.neighboursearch.LinearNNSearch;

public class ParametroEkorketa {
	
	static int k = 1;
	static int w = 0;
	static int d = 0;
	static Tag tag;
	static double maxPrecision = Double.MIN_VALUE;
	
	public static void main(String[] args) throws Exception {
		
		// k-NN algoritmoa erabili
		//Parametro ekorketa gaitasuna
		
		
		//balance-scale.arff --> klasea nominala
		//wine.arff --> klasea nominala
		
		//Ebaluazio teknika --> 10-FCV
		
		//1.Parametro ekorketa --> sacamos los parametros
		//2.Eratu sailkatzailea eta gorde (SerializationHelper)modeloa gorde
		//3. Kalitate estimazioa
		 
		
		
		DataSource source = new DataSource(args[1]);
		Instances data = source.getDataSet();
		data.setClassIndex(0);
		
		System.out.println("Instantzia totalak " + data.numInstances());
		System.out.println("Atributu totalak :" + data.numAttributes());
		int kkopurua = data.numInstances();
		
		int min = Integer.MAX_VALUE;
		int minClassIndex = 0;
		for (int i = 0; i < data.numClasses(); i++) {
			int x = data.attributeStats(data.classIndex()).nominalCounts[i];
			System.out.println(data.attribute(data.classIndex()).value(i) + "-->" 
			+ x + " instantzia kopurua" );
			
			if(x < min){
				min = x;
				minClassIndex = i;
			}
		}
		
		
		System.out.println("Klase minoritarioa: " + data.attribute(data.classIndex()).value(minClassIndex));
		System.out.println();
		System.out.println("---------------kNN Parametro Ekorketa-----------");
		
		//SelectedTag(int tagID, Tag[] tags)
//		Parameters:
//			tagID - the id of the selected tag.
//			tags - an array containing the possible valid Tags.
		
		
//		static Tag[]	TAGS_WEIGHTING
//		possible instance weighting methods.
//		static int	WEIGHT_INVERSE
//		weight by 1/distance.
//		static int	WEIGHT_NONE
//		no weighting.
//		static int	WEIGHT_SIMILARITY
//		weight by 1-distance.
		
		LinearNNSearch[] distantziak = distantziak();
		System.out.println("Distantziak kargatuta");
		SelectedTag[] tags = new SelectedTag[]{new SelectedTag(WEIGHT_NONE, TAGS_WEIGHTING), new SelectedTag(WEIGHT_INVERSE, TAGS_WEIGHTING), new SelectedTag(WEIGHT_SIMILARITY, TAGS_WEIGHTING)};
		System.out.println("Tags kargatuta");
		System.out.println("K, kopurua :" + kkopurua);
		
		
		for (int i = 1; i < kkopurua; i++) {
			IBk model = new IBk();
			
			model.setKNN(i);
			
			for (int j = 0; j < distantziak.length; j++) {
				model.setNearestNeighbourSearchAlgorithm(distantziak[j]);
				
				
				for (int l = 0; l < tags.length; l++) {
					model.setDistanceWeighting(tags[l]);
					//model.buildClassifier(data); ez da beharrezkoa crossValidation erabilita
					
					
					Evaluation eval = new Evaluation(data);
					eval.crossValidateModel(model, data, 3, new Random(3));
					
					if(eval.fMeasure(minClassIndex) > maxPrecision){
						maxPrecision = eval.fMeasure(minClassIndex);
						k=i;
						d=j;
						SelectedTag tagOna = model.getDistanceWeighting();
						tag = tagOna.getSelectedTag();
						w=l;
					}
					
				}
			}
			
		}
		
		System.out.println("f-measure maximoa -> " + maxPrecision);
        System.out.println("K hoberena -> " + k);
        System.out.println("Distantzia mota hoberena -> " + motaLortu(d));
        System.out.println("Tag -> " + tag);
		
		//Modeloa eratu sortu parametro onenekin
        IBk model = new IBk();
		
        model.setKNN(k);
        model.setNearestNeighbourSearchAlgorithm(distantziak[d]);
        model.setDistanceWeighting(tags[w]);
        model.buildClassifier(data);
        
        
        System.out.println("\nModeloa idatzi");
        modeloa_idatzi(args[2], model);
        System.out.println("\nModeloa irakurri");
        modeloa_irakurri(args[2]);
		
	}
	
	
	private static String motaLortu(int d){
		String mota = "";
		
		if (d==0) {
			mota="EuclideanDistance";
		}
		else if(d==1){
			mota="ManhattanDistance";
		}
		else {
				mota="MinkowskiDistance";
			}
		
			
		return mota;
	}


	private static LinearNNSearch[]  distantziak() throws Exception{
		
		
		LinearNNSearch euclideanDistance = new LinearNNSearch();
		euclideanDistance.setDistanceFunction(new EuclideanDistance());
		
		LinearNNSearch manhattanDistance = new LinearNNSearch();
		euclideanDistance.setDistanceFunction(new ManhattanDistance());
		
		LinearNNSearch minkowskiDistance = new LinearNNSearch();
		euclideanDistance.setDistanceFunction(new MinkowskiDistance());
		
		
		
		return new LinearNNSearch[]{euclideanDistance, manhattanDistance, minkowskiDistance};
		
	}

	private static void modeloa_idatzi(String direktorio, Classifier modeloa) throws Exception{
		SerializationHelper.write(direktorio, modeloa);
	}
	
	private static void modeloa_irakurri(String direktorioa) throws Exception{
		SerializationHelper.read(direktorioa);
		System.out.println("> Modeloa kargatu da: ");
		System.out.println("..................................................");
		Classifier sailkatzaile = (Classifier) SerializationHelper.read(direktorioa);
		System.out.println(sailkatzaile.toString());
	}
	
	
	

	private static void fitxategia_idatzi(Evaluation ev, String direktorio)  {
		
		try {
			
			FileWriter file = new FileWriter(direktorio);
			
			PrintWriter pw = new PrintWriter(file);
			
			pw.println("Direktorioa -->" + direktorio);

			pw.println("Nahasmen matrizea -->" + ev.toMatrixString());
			
			pw.close();
		} catch (Exception e) {
			
			e.printStackTrace();
		}
	}
	
	
	
	
	
	
}
