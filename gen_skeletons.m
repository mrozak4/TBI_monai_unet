files = dir(fullfile('james_preds','*_seg_warped.mat'));
for k=progress(1:length(files))
    basefilename = files(k).name;
    file =  fullfile('james_preds',basefilename);
    %InfoImage = imfinfo(tiffile);
    %mImage = InfoImage(1).Width;
    %nImage = InfoImage(1).Height;
    %NumberImages=length(InfoImage);
    %FinalImage = zeros(nImage,mImage,NumberImages,'uint8');

    %TifLink = Tiff(tiffile,'r');
    %for i=1:NumberImages
    %    TifLink.setDirectory(i);
    %    FinalImage(:,:,i)=TifLink.read();
    %end
    %TifLink.close();
    load(file);
    FilteredImage = bwskel(imbinarize(FinalImage),'MinBranchLength', 50);
    outputFileName = fullfile('james_preds',replace(basefilename,'_seg_warped.mat','_skel_warped.mat'));
    save(outputFileName,'FilteredImage','-mat')
    %for K=1:length(FilteredImage(1, 1, :))
    %    imwrite(FilteredImage(:, :, K), outputFileName, 'WriteMode', 'append',  'Compression','none');
    %end
end

% run through bwareopen function to segment
