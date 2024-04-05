rng('shuffle');
rngState = rng;
deltaSeed = uint32(feature('getpid'));
seed = rngState.Seed + deltaSeed;
rng(seed);
files = dir(fullfile('matt_raw_warped_single_upsampled_seg','*_seg_warped_single_sing.mat'));
for k=progress(randperm(length(files)))
    basefilename = files(k).name;
    if  ~isfile(fullfile('matt_raw_warped_single_upsampled_seg',replace(basefilename,'_seg_warped_single_sing.mat','_skel_warped_single_sing.mat')))
        file =  fullfile('matt_raw_warped_single_upsampled_seg',basefilename);
        load(file);
        FilteredImage = bwskel(imfill(imbinarize(FinalImage),'holes'),'MinBranchLength', 40);
        outputFileName = fullfile('matt_raw_warped_single_upsampled_seg',replace(basefilename,'_seg_warped_single_sing.mat','_skel_warped_single_sing.mat'));
        save(outputFileName,'FilteredImage','-mat')
    end
    %for K=1:length(FilteredImage(1, 1, :))
    %    imwrite(FilteredImage(:, :, K), outputFileName, 'WriteMode', 'append',  'Compression','none');
    %end
end

% run through bwareopen function to segment
