package rz;

import java.awt.BorderLayout;
import java.awt.EventQueue;

import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.border.EmptyBorder;
import javax.swing.JLabel;
import javax.swing.JTextField;
import javax.swing.JButton;
import javax.swing.JTextArea;
import javax.swing.JScrollBar;
import javax.swing.JScrollPane;

import java.awt.event.ActionListener;
import java.awt.event.ActionEvent;
import java.io.*;
import java.text.DecimalFormat;
import java.util.Scanner;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import javax.swing.ScrollPaneConstants;
import java.awt.Font;

public class rz extends JFrame {

	private JPanel contentPane;
	private static JTextField textField;
	static double E=0,EY=0;//E为关键词计数器，EY为用户比例计数器
	static int[] yhdang=new int [3];
	static String xg;
	static String [] classes = new String[] {"军事","科技","体育","旅游","食物","史政","文艺","社会","娱乐","美容"};
	private JTextField textField_1;
	private JTextField textField_2;
	private JTextField textField_3;
	private JTextField textField_4;
	private JTextField textField_5;
	
	
	
	
	/**
	 * Launch the application.
	 */
public static void readpart(String path,String a[][],JTextArea textAreas) {              //把普通电网用户行为模型存入二维数组a中
		
		try {
			String encoding = "GBK";
			File file = new File(path);
			if(file.isFile()&&file.exists()) {
				 FileInputStream stream = new FileInputStream(file);
 				 int j=1,i=1;
				 for(i=1;i<=10;i++) {
					 int pos = 1;//从第几个字节后开始读
					 int len = 4;//读几个字节
					 stream.skip(pos); //跳过之前的字节数
					 byte[] b = new byte[len];
					 stream.read(b);
					 if((new String(b)) != null) {
						 a[0][j] = new String(b);
						 j++;
						 }
				 }
				 int r,s;
				 for(r=1;r<=12;r++) {
				 int pos = 2,len = 4;
				 stream.skip(pos);
				 byte[] b = new byte[len];
				 stream.read(b);
				 if((new String(b)) !=null) {
					 a[r][0] = new String(b);
				 }
				 for(s=1;s<=10;s++) {
					 pos = 1;len = 2;
					 stream.skip(pos);
					 byte[] c = new byte[len];
					 stream.read(c);
					 if((new String(c)) != null) {
						 a[r][s] = new String(c);
					 }
				 }
				 }
			     stream.close();
			}
		} catch (Exception e) {
			textAreas.append("模型读取内容出错+\r\n");
			e.printStackTrace();
		}
	}
	
public static void Regex(double b[],String a[][],String path,JTextArea textArea) {                   //分析日志，找出里面关键词分别所占的百分比并存入一维数组b中
	try {
        String encoding="GBK";
        
		File file = new File(path);     //创建文件对象
		                
        DecimalFormat df = new DecimalFormat("0.00");
        String s;
        if(file.isFile() && file.exists()) { //判断文件是否存在
        	int j=1;//j是用电情况的序号，k用来统计日志的条数
        	String[] str=new String[10];
            double g;
            for(j=1;j<11;j++) {
            	s = a[0][j];
            	
            	
            	if(s.equals("军事")) {           //把每类关键词都分别放在txt文件里，然后将关键词文件与用户日志文件进行比对，分别找出每类关键词所占百分比并存入数组b中
            		File file1 = new File("src/mywork/关键词/军事.txt");
        			InputStreamReader read1 = new InputStreamReader(
                            new FileInputStream(file1),encoding);//考虑到编码格式
                            BufferedReader bufferedReader1 = new BufferedReader(read1);
                    String lineTxt1 = null;
                    int i = 0;                               //i表示关键词的个数
                    while((lineTxt1 = bufferedReader1.readLine()) != null) {
                    	Pattern p = Pattern.compile(lineTxt1);
                    	InputStreamReader read = new InputStreamReader(
                    			new FileInputStream(file),encoding);//考虑到编码格式
                        		BufferedReader bufferedReader = new BufferedReader(read);
                        String lineTxt = null;
                        
                    	while((lineTxt = bufferedReader.readLine()) != null) {
                    		
                    		Matcher m = p.matcher(lineTxt);
                    		if(m.find()){
                    			i++;
                    		}
                    	}
                    }
                    b[0] = i;
            	}
            	else if(s.equals("科技")) {
            		File file1 = new File("src/mywork/关键词/科技.txt");
        			InputStreamReader read1 = new InputStreamReader(
                            new FileInputStream(file1),encoding);//考虑到编码格式
                            BufferedReader bufferedReader1 = new BufferedReader(read1);
                    String lineTxt1 = null;
                    int i = 0;                               //i表示关键词的个数
                    while((lineTxt1 = bufferedReader1.readLine()) != null) {
                    	Pattern p = Pattern.compile(lineTxt1);
                    	InputStreamReader read = new InputStreamReader(
                    			new FileInputStream(file),encoding);//考虑到编码格式
                        		BufferedReader bufferedReader = new BufferedReader(read);
                        String lineTxt = null;
                       
                    	while((lineTxt = bufferedReader.readLine()) != null) {
                    		
                    		Matcher m = p.matcher(lineTxt);
                    		if(m.find()){
                    			i++;
                    		}
                    	}
                    }
                    b[1] = i;
            	}
            	else if(s.equals("体育")) {
            		File file1 = new File("src/mywork/关键词/体育.txt");
        			InputStreamReader read1 = new InputStreamReader(
                            new FileInputStream(file1),encoding);//考虑到编码格式
                            BufferedReader bufferedReader1 = new BufferedReader(read1);
                    String lineTxt1 = null;
                    int i = 0;                               //i表示关键词的个数
                    while((lineTxt1 = bufferedReader1.readLine()) != null) {
                    	Pattern p = Pattern.compile(lineTxt1);
                    	InputStreamReader read = new InputStreamReader(
                    			new FileInputStream(file),encoding);//考虑到编码格式
                        		BufferedReader bufferedReader = new BufferedReader(read);
                        String lineTxt = null;
                       
                    	while((lineTxt = bufferedReader.readLine()) != null) {
                    		
                    		Matcher m = p.matcher(lineTxt);
                    		if(m.find()){
                    			i++;
                    		}
                    	}
                    }
                    b[2] = i;
            	}
            	else if(s.equals("旅游")) {
            		File file1 = new File("src/mywork/关键词/旅游.txt");
        			InputStreamReader read1 = new InputStreamReader(
                            new FileInputStream(file1),encoding);//考虑到编码格式
                            BufferedReader bufferedReader1 = new BufferedReader(read1);
                    String lineTxt1 = null;
                    int i = 0;                               //i表示关键词的个数
                    while((lineTxt1 = bufferedReader1.readLine()) != null) {
                    	Pattern p = Pattern.compile(lineTxt1);
                    	InputStreamReader read = new InputStreamReader(
                    			new FileInputStream(file),encoding);//考虑到编码格式
                        		BufferedReader bufferedReader = new BufferedReader(read);
                        String lineTxt = null;
                       
                    	while((lineTxt = bufferedReader.readLine()) != null) {
                    		
                    		Matcher m = p.matcher(lineTxt);
                    		if(m.find()){
                    			i++;
                    		}
                    	}
                    }
                    b[3] = i;
            	}
            	else if(s.equals("季节性用电")) {
            		File file1 = new File("src/mywork/关键词/季节性用电.txt");
        			InputStreamReader read1 = new InputStreamReader(
                            new FileInputStream(file1),encoding);//考虑到编码格式
                            BufferedReader bufferedReader1 = new BufferedReader(read1);
                    String lineTxt1 = null;
                    int i = 0;                               //i表示关键词的个数
                    while((lineTxt1 = bufferedReader1.readLine()) != null) {
                    	Pattern p = Pattern.compile(lineTxt1);
                    	InputStreamReader read = new InputStreamReader(
                    			new FileInputStream(file),encoding);//考虑到编码格式
                        		BufferedReader bufferedReader = new BufferedReader(read);
                        String lineTxt = null;
                       
                    	while((lineTxt = bufferedReader.readLine()) != null) {
                    		
                    		Matcher m = p.matcher(lineTxt);
                    		if(m.find()){
                    			i++;
                    		}
                    	}
                    }
                    b[4] = i;
            	}
            	else if(s.equals("窃电")) {
            		File file1 = new File("src/mywork/关键词/窃电.txt");
        			InputStreamReader read1 = new InputStreamReader(
                            new FileInputStream(file1),encoding);//考虑到编码格式
                            BufferedReader bufferedReader1 = new BufferedReader(read1);
                    String lineTxt1 = null;
                    int i = 0;                               //i表示关键词的个数
                    while((lineTxt1 = bufferedReader1.readLine()) != null) {
                    	Pattern p = Pattern.compile(lineTxt1);
                    	InputStreamReader read = new InputStreamReader(
                    			new FileInputStream(file),encoding);//考虑到编码格式
                        		BufferedReader bufferedReader = new BufferedReader(read);
                        String lineTxt = null;
                       
                    	while((lineTxt = bufferedReader.readLine()) != null) {
                    		
                    		Matcher m = p.matcher(lineTxt);
                    		if(m.find()){
                    			i++;
                    		}
                    	}
                    }
                    b[5] = i;
            	}
            	else if(s.equals("天气情况用电")) {
            		File file1 = new File("src/mywork/关键词/天气情况用电.txt");
        			InputStreamReader read1 = new InputStreamReader(
                            new FileInputStream(file1),encoding);//考虑到编码格式
                            BufferedReader bufferedReader1 = new BufferedReader(read1);
                    String lineTxt1 = null;
                    int i = 0;                               //i表示关键词的个数
                    while((lineTxt1 = bufferedReader1.readLine()) != null) {
                    	Pattern p = Pattern.compile(lineTxt1);
                    	InputStreamReader read = new InputStreamReader(
                    			new FileInputStream(file),encoding);//考虑到编码格式
                        		BufferedReader bufferedReader = new BufferedReader(read);
                        String lineTxt = null;
                       
                    	while((lineTxt = bufferedReader.readLine()) != null) {
                    		
                    		Matcher m = p.matcher(lineTxt);
                    		if(m.find()){
                    			i++;
                    		}
                    	}
                    }
                    b[6] = i;
            	}
            	else if(s.equals("周期用电")) {
            		File file1 = new File("src/mywork/关键词/周期用电.txt");
        			InputStreamReader read1 = new InputStreamReader(
                            new FileInputStream(file1),encoding);//考虑到编码格式
                            BufferedReader bufferedReader1 = new BufferedReader(read1);
                    String lineTxt1 = null;
                    int i = 0;                               //i表示关键词的个数
                    while((lineTxt1 = bufferedReader1.readLine()) != null) {
                    	Pattern p = Pattern.compile(lineTxt1);
                    	InputStreamReader read = new InputStreamReader(
                    			new FileInputStream(file),encoding);//考虑到编码格式
                        		BufferedReader bufferedReader = new BufferedReader(read);
                        String lineTxt = null;
                       
                    	while((lineTxt = bufferedReader.readLine()) != null) {
                    		
                    		Matcher m = p.matcher(lineTxt);
                    		if(m.find()){
                    			i++;
                    		}
                    	}
                    }
                    b[7] = i;
            	}
            	else if(s.equals("日内分时用电")) {
            		File file1 = new File("src/mywork/关键词/日内分时用电.txt");
        			InputStreamReader read1 = new InputStreamReader(
                            new FileInputStream(file1),encoding);//考虑到编码格式
                            BufferedReader bufferedReader1 = new BufferedReader(read1);
                    String lineTxt1 = null;
                    int i = 0;                               //i表示关键词的个数
                    while((lineTxt1 = bufferedReader1.readLine()) != null) {
                    	Pattern p = Pattern.compile(lineTxt1);
                    	InputStreamReader read = new InputStreamReader(
                    			new FileInputStream(file),encoding);//考虑到编码格式
                        		BufferedReader bufferedReader = new BufferedReader(read);
                        String lineTxt = null;
                       
                    	while((lineTxt = bufferedReader.readLine()) != null) {
                    	
                    		Matcher m = p.matcher(lineTxt);
                    		if(m.find()){
                    			i++;
                    		}
                    	}
                    }
                    b[8] = i;
            	}
            	else if(s.equals("节假日用电")) {
            		File file1 = new File("src/mywork/关键词/节假日用电.txt");
        			InputStreamReader read1 = new InputStreamReader(
                            new FileInputStream(file1),encoding);//考虑到编码格式
                            BufferedReader bufferedReader1 = new BufferedReader(read1);
                    String lineTxt1 = null;
                    int i = 0;                               //i表示关键词的个数
                    while((lineTxt1 = bufferedReader1.readLine()) != null) {
                    	Pattern p = Pattern.compile(lineTxt1);
                    	InputStreamReader read = new InputStreamReader(
                    			new FileInputStream(file),encoding);//考虑到编码格式
                        		BufferedReader bufferedReader = new BufferedReader(read);
                        String lineTxt = null;
                       
                    	while((lineTxt = bufferedReader.readLine()) != null) {
                    		
                    		Matcher m = p.matcher(lineTxt);
                    		if(m.find()){
                    			i++;
                    		}
                    	}
                    }
                    b[9] = i;
            	}
            
            	
            }
            
             //用户日志总数k
            for(j=0;j<b.length;j++)
            {
            	E=E+b[j];
            }
            for(j=0;j<10;j++) {
            	g=(((double)b[j])/E*100);           //用户的各个搜索关键字在日志总数k中所占的百分比
            	b[j]=g;
            }
            for(j=0;j<b.length;j++)
            	str[j]=""+b[j];
            for(j=0;j<str.length;j++)
            	{
            	textArea.append(classes[j]+str[j]);
            	textArea.append("\r\n");
            	}
            
            
            
            for(j=0;j<=4;j++)//理科档
            	EY=EY+b[j];
            
            if(EY>=0&&EY<33)
            	EY=1;
            else
            if(EY>=33&&EY<66)
            	EY=2;
            else
            if(EY>=66&&EY<=100)
            	EY=3;
            yhdang[1]=(int)EY;
            EY=0;
            
            for(j=5;j<=9;j++)//文科档
            	EY=EY+b[j];
            
            if(EY>=0&&EY<33)
            	EY=1;
            else
            if(EY>=33&&EY<66)
            	EY=2;
            else
            if(EY>=66&&EY<=100)
            	EY=3;
            yhdang[2]=(int)EY;
            E=0;EY=0;
        }
        else
        	textField.setText("此文件不存在！");
	} catch (Exception e) {
		textField.setText("读取文件内容出错");
    	e.printStackTrace();
	}
}

public static void core(String a[][],double arr[],double srr[][]) {           //求用户搜索频度向量与标准搜索频度向量的夹角的余弦值
	DecimalFormat df = new DecimalFormat("0.00000");
	int i,j;
	double s,p,q=0,x,y,d;
	for(i=0;i<10;i++){
		q = q + arr[i]*arr[i];                                            //用户搜索频度向量的模的平方q
	}
	for(i=1;i<13;i++){
		s=0;p=0;
		for(j=1;j<11;j++){
			s = s + arr[j-1]*(Integer.parseInt(a[i][j]));                 //求两个向量的向量积s
			p = p + (Integer.parseInt(a[i][j]))*(Integer.parseInt(a[i][j]));     //标准搜索频度向量的模的平方p
		}
		x = Math.sqrt(q) * Math.sqrt(p);
		y = s/x;                                                          //两向量夹角余弦值y
		srr[1][i-1] = y;
		srr[0][i-1] = i;
	}
	for(i=0;i<11;i++) {
		for(j=i+1;j<12;j++) {
			if(srr[1][i] > srr[1][j]) {
				d = srr[1][i];
				srr[1][i] = srr[1][j];
				srr[1][j] = d;
				d = srr[0][i];
				srr[0][i] = srr[0][j];
				srr[0][j] = d;
			}
		}
	}
	
}

public static void core0(String s,JTextArea textArea,int n,String p) {   
	
	int i;
	try {
		String encoding="GBK";
		File file = new File(p);
		if(file.isFile() && file.exists()) {    //判断文件是否存在
			InputStreamReader read = new InputStreamReader(new FileInputStream(file),encoding);//考虑到编码格式
                    
                    BufferedReader bufferedReader = new BufferedReader(read);
                    
                   String lineTxt=null;
                   String linetxt=null;
                   lineTxt = bufferedReader.readLine();
                  while((lineTxt) != null){
                	lineTxt = bufferedReader.readLine();
                     if(s.equals(lineTxt)) {
                    		for(i=0;i<n;i++) {
                    			linetxt = bufferedReader.readLine();
                    			textArea.append(linetxt+"\r\n");
                    			
                    			
                    		}
                    	}
                    }
		}
	} catch (Exception e) {
		textArea.append("读取文件内容出错");
    	e.printStackTrace();
	}
	
}
public static void readTxtFile(String filePath,JTextArea textAreas)
{
    try 
    {
        int n;    
    	String encoding="GBK";
            File file=new File(filePath);
            if(file.isFile() && file.exists())
            { //判断文件是否存在
                InputStreamReader read = new InputStreamReader(
                new FileInputStream(file),encoding);//考虑到编码格式
                BufferedReader bufferedReader = new BufferedReader(read);
                String lineTxt = null;
                for(n=0;n<10;n++)
                {
                	lineTxt = bufferedReader.readLine();
                    textAreas.append(lineTxt+"\r\n");
                }    
                read.close();
                
                
            }
            else
            {
    	      textAreas.append("找不到指定的文件");
            }
    } 
    catch (Exception e) {
    	textAreas.append("读取文件内容出错");
        e.printStackTrace();
    }
 
}
public void WriteStringToFile(String filePath, String text) {  
    try {  
        File file = new File(filePath);  
        PrintStream ps = new PrintStream(new FileOutputStream(file));  
        ps.append(text);// 在已有的基础上添加字符串  
    } catch (FileNotFoundException e) {  
        // TODO Auto-generated catch block  
        e.printStackTrace();  
    }  
}  
	public static void main(String[] args) {
		EventQueue.invokeLater(new Runnable() {
			public void run() {
				try {
					rz frame = new rz();
					frame.setVisible(true);
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		});
	}

	public static int appearNumber(String srcText, String findText) {
	    int count = 0;
	    Pattern p = Pattern.compile(findText);
	    Matcher m = p.matcher(srcText);
	    while (m.find()) {
	        count++;
	    }
	    return count;
	}	
	public rz() {
		int i;
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		setBounds(100, 100, 1698, 983);
		contentPane = new JPanel();
		contentPane.setBorder(new EmptyBorder(5, 5, 5, 5));
		setContentPane(contentPane);
		contentPane.setLayout(null);
		
		JLabel label = new JLabel("\u8BF7\u8F93\u5165\u7528\u6237\u65E5\u5FD7\u6587\u4EF6\u5939\u7684\u7EDD\u5BF9\u8DEF\u5F84\uFF1A");
		label.setFont(new Font("宋体", Font.PLAIN, 20));
		label.setBounds(14, 13, 344, 18);
		contentPane.add(label);
		
		textField = new JTextField();
		textField.setBounds(14, 44, 802, 24);
		contentPane.add(textField);
		textField.setColumns(10);
		
		JButton button = new JButton("\u5F00\u59CB\u5206\u6790");
		button.setFont(new Font("宋体", Font.PLAIN, 20));
		button.setBounds(861, 43, 123, 27);
		contentPane.add(button);
		JButton button_1 = new JButton("\u91CD\u65B0\u6D4B\u8BD5");
		button_1.setFont(new Font("宋体", Font.PLAIN, 20));
		button_1.setBounds(1327, 43, 123, 27);
		contentPane.add(button_1);
		
		JLabel lblNewLabel = new JLabel("\u7528\u6237\u6027\u683C\u5206\u6790\uFF1A");
		lblNewLabel.setFont(new Font("宋体", Font.PLAIN, 20));
		lblNewLabel.setBounds(855, 112, 156, 18);
		contentPane.add(lblNewLabel);
		
		JLabel label_2 = new JLabel("\u7528\u6237\u884C\u4E3A\u9884\u6D4B\uFF1A");
		label_2.setFont(new Font("宋体", Font.PLAIN, 20));
		label_2.setBounds(1147, 112, 156, 18);
		contentPane.add(label_2);
		
		JLabel label_12 = new JLabel("\u7528\u6237\u641C\u7D22\u5173\u952E\u8BCD\u6BD4\u4F8B\uFF1A");
		label_12.setFont(new Font("宋体", Font.PLAIN, 20));
		label_12.setBounds(603, 112, 213, 18);
		contentPane.add(label_12);
		
		
		JLabel label_13 = new JLabel("\u7528\u6237\u65E5\u5FD7\uFF1A");
		label_13.setFont(new Font("宋体", Font.PLAIN, 20));
		label_13.setBounds(15, 112, 107, 18);
		contentPane.add(label_13);
		
		JScrollPane scrollPane = new JScrollPane();
		scrollPane.setBounds(15, 162, 551, 733);
		scrollPane.setVerticalScrollBarPolicy(ScrollPaneConstants.VERTICAL_SCROLLBAR_ALWAYS);
		scrollPane.setHorizontalScrollBarPolicy(ScrollPaneConstants.HORIZONTAL_SCROLLBAR_ALWAYS);
		contentPane.add(scrollPane);
		
		final JTextArea textAreas = new JTextArea();
		scrollPane.setViewportView(textAreas);
		
		JScrollPane scrollPane_1 = new JScrollPane();
		scrollPane_1.setVerticalScrollBarPolicy(ScrollPaneConstants.VERTICAL_SCROLLBAR_ALWAYS);
		scrollPane_1.setHorizontalScrollBarPolicy(ScrollPaneConstants.HORIZONTAL_SCROLLBAR_ALWAYS);
		scrollPane_1.setBounds(861, 162, 251, 733);
		contentPane.add(scrollPane_1);
		
		final JTextArea textArea_1 = new JTextArea();
		scrollPane_1.setViewportView(textArea_1);
		
		
		scrollPane_1.setViewportView(textArea_1);
		
		JScrollPane scrollPane_3 = new JScrollPane();
		scrollPane_3.setVerticalScrollBarPolicy(ScrollPaneConstants.VERTICAL_SCROLLBAR_ALWAYS);
		scrollPane_3.setHorizontalScrollBarPolicy(ScrollPaneConstants.HORIZONTAL_SCROLLBAR_ALWAYS);
		scrollPane_3.setBounds(613, 162, 203, 733);
		contentPane.add(scrollPane_3);
		
		final JTextArea textArea = new JTextArea();
		scrollPane_3.setViewportView(textArea);
		
		JScrollPane scrollPane_2 = new JScrollPane();
		scrollPane_2.setVerticalScrollBarPolicy(ScrollPaneConstants.VERTICAL_SCROLLBAR_ALWAYS);
		scrollPane_2.setHorizontalScrollBarPolicy(ScrollPaneConstants.HORIZONTAL_SCROLLBAR_ALWAYS);
		scrollPane_2.setBounds(1149, 162, 322, 733);
		contentPane.add(scrollPane_2);
		
		final JTextArea textArea_2_1 = new JTextArea();
		scrollPane_2.setViewportView(textArea_2_1);
		
		JLabel label_3 = new JLabel("\u7ED3\u679C\u7EDF\u8BA1\uFF1A(\u4EBA\u6570)");
		label_3.setFont(new Font("宋体", Font.PLAIN, 20));
		label_3.setBounds(1495, 111, 166, 21);
		contentPane.add(label_3);
		
		JLabel label_1 = new JLabel("\u79EF\u6781\u6027\u683C\uFF1A");
		label_1.setFont(new Font("宋体", Font.PLAIN, 20));
		label_1.setBounds(1520, 196, 113, 21);
		contentPane.add(label_1);
		
		JLabel label_4 = new JLabel("\u4E2D\u7EA7\u6027\u683C\uFF1A");
		label_4.setFont(new Font("宋体", Font.PLAIN, 20));
		label_4.setBounds(1520, 244, 107, 21);
		contentPane.add(label_4);
		
		JLabel label_5 = new JLabel("\u6D88\u6781\u6027\u683C\uFF1A");
		label_5.setFont(new Font("宋体", Font.PLAIN, 20));
		label_5.setBounds(1520, 290, 107, 21);
		contentPane.add(label_5);
		
		JLabel label_6 = new JLabel("\u6B63\u5E38\u7528\u6237\uFF1A");
		label_6.setFont(new Font("宋体", Font.PLAIN, 20));
		label_6.setBounds(1520, 356, 113, 21);
		contentPane.add(label_6);
		
		JLabel label_7 = new JLabel("\u5F02\u5E38\u7528\u6237\uFF1A");
		label_7.setFont(new Font("宋体", Font.PLAIN, 20));
		label_7.setBounds(1520, 415, 107, 21);
		contentPane.add(label_7);
		
		textField_1 = new JTextField();
		textField_1.setBounds(1612, 193, 49, 27);
		contentPane.add(textField_1);
		textField_1.setColumns(10);
		
		textField_2 = new JTextField();
		textField_2.setBounds(1612, 241, 49, 27);
		contentPane.add(textField_2);
		textField_2.setColumns(10);
		
		textField_3 = new JTextField();
		textField_3.setBounds(1612, 287, 49, 27);
		contentPane.add(textField_3);
		textField_3.setColumns(10);
		
		textField_4 = new JTextField();
		textField_4.setBounds(1612, 353, 49, 27);
		contentPane.add(textField_4);
		textField_4.setColumns(10);
		
		textField_5 = new JTextField();
		textField_5.setBounds(1612, 415, 49, 27);
		contentPane.add(textField_5);
		textField_5.setColumns(10);
		
		JButton button_2 = new JButton("\u4FDD\u5B58\u7ED3\u679C");
		button_2.setFont(new Font("宋体", Font.PLAIN, 20));
		button_2.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				String sumpath="src/mywork/sum.txt";
				String text=textAreas.getText()+textArea.getText()+textArea_1.getText()+textArea_2_1.getText();
			     WriteStringToFile(sumpath,text);
				
			}
		});
		button_2.setBounds(1018, 42, 123, 29);
		contentPane.add(button_2);
		
		JButton button_3 = new JButton("\u5BFC\u51FA\u7ED3\u679C");
		button_3.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				try {   
				    Process process = Runtime.getRuntime().exec(  
				    "cmd.exe  /c notepad ./src/mywork/sum.txt");  
				} catch (Exception e1) {  
				e1.printStackTrace();  
				}  
			}
		});
		button_3.setFont(new Font("宋体", Font.PLAIN, 20));
		button_3.setBounds(1170, 42, 123, 29);
		contentPane.add(button_3);
		
		button_1.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent arg0) {
				textField.setText("");
				textArea_1.setText("");
				textArea_2_1.setText("");
				textArea.setText("");
				textAreas.setText("");
				textField_1.setText("");
				textField_2.setText("");
				textField_3.setText("");
				textField_4.setText("");
				textField_5.setText("");
			}
		});
		
		
		
		button.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent arg0) {//主函数
				int i,j,k,p1,t=0,sy = 0,num;
				// TODO Auto-generated method stub
				String[][] a = new String[13][11];
				String p,path,srcText,findText;
				double[] b = new double[10];                           //把用户的搜索频度向量存入一维数组b中
				double[][] c = new double[2][12];  
				int [] A=new int [3];//游标
				int [][] B=new int [27][4];//索引表
				
				
				for(i=0;i<27;i++)//构建索引表
			     {
				     B[i][3]=i+1;
			     }

			     for(i=0;i<3;i++)//27中组合
			    {
			         A[0]=i+1;
			        for(j=0;j<3;j++)
				   {
					   A[1]=j+1;
					   {
						for(k=0;k<3;k++)
							{
								A[2]=k+1;
						        
									B[t][0]=A[0];
									B[t][1]=A[1];
									B[t][2]=A[2];
									t++;
							
							}    
							
						}			
					}	
					
			   }
				
				//用户搜索频度向量与标准搜索频度向量的夹角的余弦值存入二维数组c中
				readpart("src/mywork/性格模型.txt",a,textAreas);                     //把星座模型存入二维数组a中  
				
				//开始分析日志
				String path1 = textField.getText();
				for(p1=1;p1<=100;p1++)
				{
					
					System.out.printf("第%d篇日志开始处理\n", p1);
					path=path1+ "\\"+ String.valueOf(p1)+ ".txt";
					textAreas.append("用户"+ String.valueOf(p1)+"\r\n");
					textArea.append("用户"+ String.valueOf(p1)+"\r\n");
					textArea_1.append("用户"+ String.valueOf(p1)+"\r\n");
					textArea_2_1.append("用户"+ String.valueOf(p1)+"\r\n");
					Regex(b,a,path,textArea);
					core(a,b,c);
					xg=a[(int)c[0][11]][0];
					if(xg.equals("责任")==true||xg.equals("兴趣")==true||xg.equals("生活")==true||xg.equals("家庭")==true)
			        	yhdang[0]=3;
			        else
			        if(xg.equals("上进")==true||xg.equals("实际")==true||xg.equals("争强")==true||xg.equals("自我")==true)
			        	yhdang[0]=2;
			        else
			        if(xg.equals("自弱")==true|xg.equals("压差")==true||xg.equals("不沟")==true||xg.equals("经不")==true)
			        	yhdang[0]=1;
					
				     for(i=0;i<27;i++)//遍历索引表
				     {
				     	if(B[i][0]==yhdang[0]&&B[i][1]==yhdang[1]&&B[i][2]==yhdang[2])
				     		{
				     		 sy=B[i][3];
				     	     break;
				     	    }
				     	
				     }			     
				     readTxtFile(path,textAreas);
				     String s = String.valueOf(sy);
				     p="src/mywork/用户类型.txt";
				     core0(s,textArea_1,1,p); 
				     p="src/mywork/户用电.txt";
				     core0(xg,textArea_1,7,p);
				     p="src/mywork/用电整体行为.txt";
				     core0(s,textArea_2_1,1,p); 
				     p="src/mywork/用电行为.txt";
				     core0(s,textArea_2_1,4,p);
				     
				     srcText =textArea_1.getText();
				     findText = "(用户日内用电)";
				     num = appearNumber(srcText, findText);
				     textField_1.setText(String.valueOf(num));
				     
				     findText = "(用户短期用电)";
				     num = appearNumber(srcText, findText);
				     textField_2.setText(String.valueOf(num));
				     
				     findText = "(用户长期用电)";
				     num = appearNumber(srcText, findText);
				     textField_3.setText(String.valueOf(num));
				     //textArea_2_1
				     srcText =textArea_2_1.getText();
				     findText = "(正常)";
				     num = appearNumber(srcText, findText);
				     textField_4.setText(String.valueOf(num));
				     
				     findText = "(异常/窃电)";
				     num = appearNumber(srcText, findText);
				     textField_5.setText(String.valueOf(num));
				}     
				     		     
			}
		});
		
	}
}
