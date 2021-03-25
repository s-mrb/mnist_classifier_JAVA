import Jama.Matrix;
import java.io.*;
import static java.lang.System.*;
import Jama.*;
import java.util.Random;

public class nn1 {
    static Random rand = new Random();

    public static Matrix get_random_matrix(int row, int col,double mean,double std_dev){
        Matrix x = new Matrix(row,col);
        double randomValue ;
        for (int i=0; i<row; i++){
            for (int j=0; j<col; j++){
                x.set(i,j,mean + rand.nextGaussian()*std_dev);
            }
        }
        return x;
    }

    static double mean(Matrix x)
    {
        int row = x.getRowDimension();
        int col = x.getColumnDimension();
        // Calculating sum
        double sum = 0;
        for (int i = 0; i < row; i++)
            for (int j = 0; j < col; j++)
                sum += x.get(i,j);
        // Returning mean
        return sum / (row * col);
    }

    static double variance(Matrix x, double m)
    {
        int row = x.getRowDimension();
        int col = x.getColumnDimension();
        int sum = 0;
        double ans=0 ;
        double temp;
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < col; j++)
            {
                temp = x.get(i,j);
                temp = (temp - m);
                temp = temp*temp;
                ans = ans + temp;
            }
        }
        ans = ans/(row*col);
        return ans;
    }

    static double std_deviation(Matrix x, double m)
    {
        double var = variance(x,m);
        double std_d = Math.sqrt(var);
        return std_d;
    }

    public static double get_accuracy(Matrix truth, Matrix pred)
    {
        int col = truth.getColumnDimension();
        int tru = 0;
        for (int c=0; c<col; c++)
        {
            if (check_label(truth, pred, c))
            {
                tru = tru + 1;
            }
        }
        return ((double) tru/col)*100;
    }

    private static boolean check_label(Matrix truth, Matrix pred, int c)
    {
        int row = truth.getRowDimension();
        double max = 0.0;
        int max_row=-1;

        for (int r=0; r<row; r++)
        {
            double val = pred.get(r,c);
            if (val>max)
            {
                max = val;
                max_row = r;
            }
        }

        if (truth.get(max_row,c)>0)
        {
            return true;
        }
        else
        {
            return false;
        }
    }


    public static double sum_matrix(Matrix x)
    {
        double sum = 0;
        int row = x.getRowDimension();
        int col = x.getColumnDimension();
        for (int r=0; r<row; r++)
        {
            for (int c=0; c<col; c++)
            {
                sum = sum +x.get(r,c);
            }
        }
        return sum;
    }

    public static int argmax(Matrix x, int col)
    {
        int row = x.getRowDimension();
        double max = 0.0;
        int max_row=-1;
        for (int r=0; r<row; r++)
        {
            double val = x.get(r,col);
            if (val>max)
            {
                max = val;
                max_row = r;
            }
        }
        return max_row;
    }

    //It takes a nx1 vector and int y and return nxy matrix
    public static Matrix expand_along_y(Matrix b,int y)
    {
        int row = b.getRowDimension();
        int col = y;
        Matrix b_expand = new Matrix(row,col);

        for (int i=0; i<row; i++)
        {
            for (int j=0; j<col; j++)
            {
                b_expand.set(i,j,b.get(i,0));
            }
        }
    return b_expand;
    }

    public static double sigmoid_f(double z)
    {
        double activation;
        activation  =  1 / (1 + Math.pow(Math.E,-z));
        return activation;
    }

    public static Matrix npsum0_keepdim(Matrix z)
    {
        int row = z.getRowDimension();
        int col = z.getColumnDimension();
        Matrix activation = new Matrix(1,col);
        double temp2=0.0;
        for (int i=0; i<col; i++)
        {
            for (int j=0; j<row; j++)
            {
                temp2 = temp2 + z.get(j,i);
            }
            activation.set(0,i,temp2);
            temp2 =0.0;
        }
        return activation;
    }

    public static Matrix npsum1_keepdim(Matrix z)
    {
        int row = z.getRowDimension();
        int col = z.getColumnDimension();
        Matrix activation = new Matrix(row,1);
        double temp2=0.0;
        for (int i=0; i<row; i++)
        {
            for (int j=0; j<col; j++)
            {
                temp2 = temp2 + z.get(i,j);

            }
            activation.set(i,0,temp2);
            temp2 =0.0;
        }

        return activation;
    }

    public static Matrix softmax_f(Matrix z)
    {
        int row = z.getRowDimension();
        int col = z.getColumnDimension();
        Matrix activation = new Matrix(row,col);

        for(int i=0; i<row; i++)
        {
            for (int j=0; j<col; j++)
            {
                activation.set(i,j,Math.pow(Math.E,z.get(i,j)));
            }
        }
        double[] deno = new double[col];
        double temp2=0.0;
        for (int i=0; i<col; i++)
        {
            for (int j=0; j<row; j++)
            {
                temp2 = temp2 + activation.get(j,i);

            }
            deno[i] = temp2;
            temp2 =0;
        }
        for ( int i=0; i<row; i++ )
        {
            for (int j=0; j<col; j++)
            {

                activation.set(i,j,activation.get(i,j)/deno[j]);
            }
        }
        return activation;
    }

    public static Matrix sigmoid_f( Matrix z )
    {
        int row = z.getRowDimension();
        int col = z.getColumnDimension();
        Matrix activation = new Matrix(row,col);
        for (int i=0; i<row; i++){
            for (int j=0; j<col; j++)
            {
                activation.set(i,j,(1 / (1 + Math.pow(Math.E,-z.get(i,j)))));
            }
        }
        return activation;
    }

    public static double get_loss_via_normF( Matrix truth, Matrix pred )
    {
        double m = truth.getColumnDimension();
        Matrix log_pred = new Matrix(10, (int)m);

        for (int i=0; i<10; i++)
        {
            for (int j=0; j<m; j++)
            {
                log_pred.set(i,j,Math.log(pred.get(i,j)));
            }
        }
        double summation = -(1/m)*( truth.arrayTimes(log_pred) ).normF();
        return summation;
    }

    public static double get_loss_via_sum(Matrix truth, Matrix pred)
    {
        double m = truth.getColumnDimension();
        Matrix log_pred = new Matrix(10, (int)m);

        for (int i=0; i<10; i++)
        {
            for (int j=0; j<m; j++)
            {
                log_pred.set( i, j, Math.log(pred.get(i,j)) );
            }
        }
        double summation = -(1/m)*sum_matrix( truth.arrayTimes(log_pred) );
        return summation;
    }


    public static void show_images( Matrix data, Matrix labels,int[] imgs2show, Matrix predictions )
    {
        System.out.print( "\n\n\n" );
        for( int indx=0; indx<imgs2show.length; indx++ )
        {
            System.out.print( "Image: " + imgs2show[indx] + "\n" );
            for ( int i = 0; i < 784; i++ )
            {
                if( data.get( i, indx ) > 0 )
                {
                    System.out.print(1);
                }
                else {
                    System.out.print(0);
                }

                if ( (i + 1) % 28 == 0 )
                {
                    System.out.print( "\n" );
                }
            }
            int v = 0;
            for ( int q = 0; q < 10; q++ )
            {
                if ( labels.get(q,imgs2show[indx]) > 0 )
                {
                    v = q;
                    break;
                }
            }
            System.out.print( "\nLabel: " + v + "\n" );
            System.out.print( "Model Prediction: " + argmax( predictions, imgs2show[indx]) + "\n\n\n\n\n" );
        }
    }

    public static void main( String[] args)
    {

        String path2csv_in_drive = args[0];
        int epoch = Integer.parseInt(args[1]);
        int lr = Integer.parseInt(args[2]);
        int stats = Integer.parseInt(args[3]);

//        String path_on_local_machine = "C:\\Users\\Mehdi\\eclipse-workspace\\we\\src\\mnist_784.csv";

        //delimiter
        final String delimiter = ",";

        //Matrices for data and labels
        //train data
        double data_xtr [][] = new double[784][60000];
        //train label
        double data_ytr [][] = new double[10][60000];
        //test data
        double data_xtes [][] = new double[784][10000];
        //test label
        double data_ytes [][] = new double[10][10000];

        //Fill with zeros
        for( int i=0; i<10 ; i++ )
        {
            for(int j=0; j<60000;j++)
            {
                data_ytr[i][j] = 0;
            }
        }
        for( int i=0; i<10; i++ )
        {
            for( int j=0; j<10000; j++ )
            {
                data_ytes[i][j] = 0;
            }
        }

        //read csv
        try {
            File file = new File( path2csv_in_drive );
            FileReader fr = new FileReader( file );
            BufferedReader br = new BufferedReader( fr );
            String line = "";
            String[] tempArr;
            int instance = 0;
            while( (line = br.readLine()) != null )
            {

                tempArr = line.split( delimiter );

                if ( line.charAt(0) =='p' )
                {
                    continue;
                }
                //In case want to read few images, replace 300k with the number of images you want to read
                if ( instance>300000 )
                {
                    exit(0 );
                }
                System.out.println( "Reading Instance:" + instance );

                if ( instance < 60000 )
                {
                    for( int i=0; i<tempArr.length-1; i++ )
                    {
                        // for scaling from -1 to 1, divide by 127.5 and subtract 1
                        data_xtr[i][instance] = (Double.parseDouble( tempArr[i] )/255.0);

                    }
                    int temp = Integer.parseInt( tempArr[ tempArr.length-1 ] );
                    data_ytr[temp][instance]= 1.0;
                    instance = instance +1;
                }
                else {
                    for( int i=0; i<tempArr.length-1; i++ )
                    {
                        data_xtes[i][ instance-60000 ] = ( Double.parseDouble( tempArr[i] )/255.0);

                    }
                    int temp = Integer.parseInt( tempArr[ tempArr.length-1 ] );
                    data_ytes[temp][ instance-60000 ] = 1.0;

                    instance = instance + 1;
                }
            }
            br.close();
            System.out.println( "\nDataset Successfully Loaded!" );

        } catch( IOException ioe )
        {
            ioe.printStackTrace();
        }


        //In order to Print images while reading or after read
        //Matrix ax = new Matrix(data_xtr);
        //Matrix ay = new Matrix(data_ytr);
        //int[] img_indx= {0,1,2,3,4,5,6,7};
        //show_images(ax,ay,img_indx,ay);
        //exit(0);


        //Hyperparameters and matrices
        double l_rate = lr;

        //r = 64 , col = 784
        Matrix W1 = get_random_matrix( 64,784,0,1 );
        Matrix b1 = get_random_matrix( 64,1,0,1 );

        Matrix W2 = get_random_matrix( 10,64,0,1 );
        Matrix b2 = get_random_matrix( 10,1,0,1 );


        Matrix X = new Matrix( data_xtr );
        data_xtr = null;
        Matrix Y = new Matrix( data_ytr );
        data_ytr = null;
        Matrix X_test = new Matrix( data_xtes );
        data_xtes = null;
        Matrix Y_test = new Matrix( data_ytes );
        data_ytes = null;

        double data_mean = mean( X );
        if (stats==1) {
            System.out.println("A1 mean: " + data_mean + " Standard Dev: " + std_deviation(X, data_mean));
        }
        //W1 shape: (64, 784)
        //b1 shape: (64, 1)
        //W2 shape: (10, 64)
        //b2 shape: (10, 1)
        //Z1 shape: (64, 60000)
        //A1 shape: (64, 60000)
        //Z2 shape: (10, 60000)
        //A2 shape: (10, 60000)
        //dZ2 shape: (10, 60000)
        //dW2 shape: (10, 64)
        //db2 shape: (10, 1)
        //dA1 shape: (64, 60000)
        //dZ1 shape: (64, 60000)
        //dW1 shape: (64, 784)
        //db1 shape: (64, 1)
        //W2 shape: (10, 64)
        //b2 shape: (10, 1)
        //W1 shape: (64, 784)
        //b1 shape: (64, 1)

      double loss = 0.0;
      int m = X.getColumnDimension();
      for( int i=0; i<epoch; i++ )
      {
          // row in W1.X
          int r = 64;
          // col in W1.X
          int c = 60000;

       Matrix Z1 = ( W1.times( X )).plusEquals( expand_along_y( b1, c ));
       Matrix A1 = sigmoid_f( Z1 );

       // row in W2.A1
       r = 10;
       // col in W2.A1
       c = 60000;

       Matrix Z2 = ( W2.times( A1 )).plusEquals( expand_along_y( b2,c ));
       Matrix A2 = softmax_f( Z2 );

       loss = get_loss_via_sum( Y, A2 );

       Matrix dZ2 = A2.minus( Y );
       Matrix dW2 = ( dZ2.times( A1.transpose() )).times( 1.0/m );
       Matrix db2 = npsum1_keepdim( dZ2 ).times( 1.0/m );
       Matrix dA1 = ( W2.transpose() ).times( dZ2 );

       Matrix sigmoid_z = sigmoid_f( Z1 );
       Matrix dZ1 = (( new Matrix( sigmoid_z.getRowDimension(), sigmoid_z.getColumnDimension(),1 ))
               .minus( sigmoid_z )).arrayTimes( sigmoid_z ).arrayTimes( dA1 );

       Matrix dW1 = dZ1.times( X.transpose() ).times( 1.0/m );

       //?test db1 ----getting zero mean and std_dv
       Matrix db1 = npsum1_keepdim( dZ1 ).times( 1.0/m );

       double db1_m = mean( db1 );

       //for debug
          if (stats==1)
          {
              System.out.println("(Initialization) db1 mean: " + db1_m + " Standard Dev: " + std_deviation(db1, db1_m));
          }

       W2 = W2.minus( dW2.times( l_rate ));
       b2 = b2.minus( db2.times( l_rate ));
       W1 = W1.minus( dW1.times( l_rate ));
       b1 = db1.minus( db1.times( l_rate ));

       if(i%2==0)
       {

           double mean = mean( W1 );
           mean = mean( W2 );
           mean = mean( b1 );
           mean = mean( b2 );
           mean = mean( A1 );

           if (stats==1)
           {

               System.out.println("W1->Mean: " + mean + " Standard Dev: " + std_deviation(W1, mean));
               System.out.println("W2->Mean: " + mean + " Standard Dev: " + std_deviation(W2, mean));
               System.out.println("b1->Mean: " + mean + " Standard Dev: " + std_deviation(b1, mean));
               System.out.println("b2->Mean: " + mean + " Standard Dev: " + std_deviation(b2, mean));
               System.out.println("A1 mean: " + mean + " Standard Dev: " + std_deviation(A2, mean));
           }

           System.out.println("Epoch: " + i + "    loss: " + loss + "\n\n");
       }
      }


      System.out.print( "\nFinal loss: " + loss );

        //TEST
        // row in W1.X_test, decided by rows in W1 = number of neurons in first hidden layer
        int r = 64;
        // col in W1.X_test, decided by cols in X_test
        int c = 10000;
        Matrix Z1 = ( W1.times( X_test )).plusEquals( expand_along_y( b1,c ));
        Matrix A1 = sigmoid_f( Z1 );

        r = 10;
        c = 10000;
        Matrix Z2 = ( W2.times( A1 )).plusEquals( expand_along_y( b2,c ));
        Matrix A2 = sigmoid_f( Z2 );

        //Use A2 and Y_test to find accuracy
        System.out.print( "\nCurrent Accuracy: " + get_accuracy( Y_test,A2 ));

        //Show test images with Model Prediction and Ground Truth Labels
        // array of image numbers you wanna see
        // first image in test set numbered 0
        int[] img_indx = { 0,1,2,3,4,5,6,7 };
        show_images( X_test, Y_test,img_indx, A2);
    }
}