% normalization_0_255
function normalized_map=normalization_0_1(input_map)
[i_width i_length dim]=size(input_map);
normalized_map=zeros(i_width,i_length,dim);
for k=1:dim
    max_input_map=input_map(1,1,k);
    min_input_map=input_map(1,1,k);
    for i=1:i_width;
        for j=1:i_length;
            if input_map(i,j,k)>max_input_map;
                max_input_map=input_map(i,j,k);
            end
            if input_map(i,j,k)<min_input_map;
                min_input_map=input_map(i,j,k);
            end
        end
    end
    if max_input_map~=min_input_map
        normalized_map(:,:,k)=(input_map(:,:,k)-min_input_map)/(max_input_map-min_input_map);
    else
        normalized_map(:,:,k)=input_map(:,:,k);
    end
end