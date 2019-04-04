package preProcess;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.LinkedHashMap;
import java.util.Map;

import org.joda.time.DateTime;
import org.joda.time.format.DateTimeFormat;
import org.joda.time.format.DateTimeFormatter;

public class OneImpressionPerLine {

	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub


		try (BufferedReader br = new BufferedReader(new FileReader(new File(args[0])))) {
			String line = br.readLine();
			PrintWriter printWriter = new PrintWriter (args[1]);
			while ((line = br.readLine()) != null) {
				String [] array = line.split("\t");
				String user = array[0];
				int	 year = Integer.parseInt(array[1].substring(2, array[1].length()));
				int week = Integer.parseInt(array[2]);
				DateTimeFormatter dtf = DateTimeFormat.forPattern("yyyy-MM-dd HH:mm:ss");
				DateTime dt = new DateTime().withYearOfCentury(year).withWeekOfWeekyear(week);
				String items  = array[3];
				String [] arrayItems = items.split(",");
				for(int i = 0 ; i < arrayItems.length ; i++){
					printWriter.println(user+"\t"+arrayItems[i]+"\t"+dtf.print(dt));
				}
			}
			printWriter.close();
		}
	}

}
