function [data, split] = split_dataset(split, data, bagsize)
% stream=RandStream('mt19937ar','Seed',2);%% generate stream for reproducibility of model
% set(stream,'Substream',1);
% RandStream.setGlobalStream(stream);
% data = ScaledMatrixByColumn(data,-1,1);
bagnum = ceil(length(split.train_label) / bagsize);
split.train_bag_idx = zeros(bagnum, 1);
for i=1:length(split.train_label),
    split.train_bag_idx(i) = mod(i, bagnum);
end
split.train_bag_idx(split.train_bag_idx==0)=bagnum;
rowrank = randperm(size(split.train_bag_idx, 1));
split.train_bag_idx = split.train_bag_idx(rowrank, :);
split.test_bag_idx = split.train_bag_idx;

split.test_label = split.train_label;

split.train_bag_prop = zeros(bagnum, 1);
for i=1:bagnum,
    for j=1:size(data,1),
        if split.train_label(j)==1 && split.train_bag_idx(j)==i,
            split.train_bag_prop(i) = split.train_bag_prop(i) + 1;
        end
    end
    split.train_bag_prop(i) = split.train_bag_prop(i)/length(find(split.train_bag_idx==i));
end
split.test_bag_prop = split.train_bag_prop;

end
