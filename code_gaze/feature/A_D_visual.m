clc
clear
A=load('A.mat');
for m=1:20
    for n=1:64
        feature=zeros(14,14);
        for i=1:14
            for j=1:14
                feature(i,j)=A.gene_features(m,n,i,j);
            end
        end
        feature=normalization_0_1(feature);
        feature=GetHeatMap(feature);
        feature=imresize(feature,[448,448]);
        name_feature=['feature_map/A/' 'A_' num2str(m) '_' num2str(n) '.png'];
        imwrite(uint8(feature*255),name_feature);
        disp(['>>gazeDestribution: batch ',num2str(m),', Channel ',num2str(n),' is being Processed ...']);
    end    
end
D=load('D.mat');
for m=1:20
    for n=1:64
        feature=zeros(14,14);
        for i=1:14
            for j=1:14
                feature(i,j)=D.gene_features(m,n,i,j);
            end
        end
        feature=normalization_0_1(feature);
        feature=GetHeatMap(feature);
        feature=imresize(feature,[448,448]);
        name_feature=['feature_map/D/' 'D_' num2str(m) '_' num2str(n) '.png'];
        imwrite(uint8(feature*255),name_feature);
        disp(['>>gazeDestribution: batch ',num2str(m),', Channel ',num2str(n),' is being Processed ...']);
    end    
end