% SHOW_IMG displays a MatLab MATRIX in the current figure, inside the 
% current axes.  
%
% range = SHOW_IMG(img, range, zoom, label, n_gray) for a gray value image
% displays the matrix img and returns the range of the displayed gray 
% values.
% 
% range = SHOW_IMG(img, range, zoom, label) for a color image displays the
% image as true-color image and returns the ranges of the 3 channels.
%
% Optional control parameters:
% 
%       range: 2-vector specifying the values of the lowest and highest 
%       gray or channel value, respectively.  Allowed values:
%           - [low, high]
%           - 'original': use original single, uint8 or uint16 values without
%           scaling, set to 'full' for floating point formats if range 
%           exceeds [0,1].
%           - 'full' (default): range = [min, max]
% 
%       zoom: display zoom factor. Allowed values:
%           - positive zoom factor
%           - 'auto' (default): zoom is chosen to fit into the current
%           axes.
%           - 'full': entire axis region is filled without space for
%           labels.
%
%       label: controls figure labeling. Allowed values:
%           - 1 (default): display range, zoom factor and size below image.
%           - 0: no labeling
%           - string: displayd additionally as title.
%       No labeling is shown for zoom = 'full'.
% 
%       n_gray: number of gray levels, default: 256 (ignored for color
%       images.
%
%   Class Support
%   -------------
%   All image formats (except sparse matrices) are allowed 
%
%
%   Copyright (C) 2009 by Matthias O. Franz. Date 2009-03-22.

function range = show_img(img, range, zoom, label, n_gray)

% read out arguments/default values
if nargin < 1
    error('no image to show_img.'); 
end
if exist('range') ~= 1
    range = 'full';
end
if exist('zoom') ~= 1
    zoom = 'auto';
end
if exist('label') ~= 1
    if strcmp(zoom,'full')
        label = 0;				% no title
    else					
        label = 1;				% print range
    end
end

% set number of grayvalues
bit_depth = get(0,'ScreenDepth'); % ckeck bit depth of screen
if bit_depth < 24
    disp(sprintf('Warning: BitDepth of Screen is only %d', bit_depth));
end
if exist('n_gray') ~= 1
    if isa(img, 'uint8')
        n_gray = 256;
    elseif isa(img, 'uint16')
        n_gray = 65536;
    else
        n_gray = 256;
    end
end
n_gray = max(n_gray, 2);

% range
ndims = length(size(img));
if strcmp(range, 'original') && (isa(img, 'double') || isa(img, 'single'))
    [mn, mx] = range_img(img);
    if mn < 0 || mx > 1        
        range = 'full'; % no natural range for float/double image
    else
        if ndims == 2
            n_gray = 256;
        end
        range = [0, 1];
        out_img = img; % copy unchanged
    end
end
if strcmp(range, 'original') % use original image range
    if isa(img, 'uint8') % uint 8 image
        if ndims == 2
            n_gray = 256;
        end
        range = [0, 255];
        out_img = img; % copy unchanged
    elseif isa(img, 'uint16') % uint16 image
        if ndims == 2
            n_gray = 65536;
        end
        range = [0, 65536];
        out_img = img; % copy unchanged
    end
else 
    out_img = double(img); % convert to double for display
    [mn, mx] = range_img(out_img);
    if mn == mx % constant image
        mx = mn + 1;
    end
    if ndims == 2 % scale image to [1.5, n_gray + 0.5]
        out_img = ((n_gray - 1.0)/(mx - mn))*(out_img - ...
        mn*ones(size(out_img))) + 1.5*ones(size(out_img));
    elseif ndims == 3
        out_img = (1.0/(mx - mn))*(out_img - mn*ones(size(out_img)));
    end
    range = [mn,mx];
end

% display image
image(out_img);
axis('off');
if ndims == 2
    colormap(gray(n_gray));
end
axis('image');

% maximize image in current window if desired
ax = gca;
old_units = get(ax,'Units');
if strcmp(zoom,'full');
    set(ax, 'Units', 'normalized');
    set(ax, 'Position', [0 0 1 1]);
    label = 0;
    zoom = 'auto';
end

% get center
set(ax,'Units','pixels');
pos = get(ax, 'Position');
ctr = pos(1:2) + pos(3:4)/2;
dims = [size(out_img, 2) size(out_img, 1)];

% determine zoom factor for maximal size
if strcmp(zoom, 'auto')
  zoom = min(pos(3:4) ./ (dims - 1));
elseif isstr(zoom)
  error(sprintf('Unknown ZOOM argument: %s', zoom));
end

% Force zoom value to be an integer, or inverse integer.
new_size = round(zoom*dims);
set(ax,'Position', [round(ctr - new_size/2), new_size])

% Restore units
set(ax, 'Units', old_units);

% figure title
if label ~= 0 % title shown
    if isstr(label) % figure title given by user
        title(label);
        h = get(gca,'Title');
        old_units = get(h,'Units');
        set(h,'Units','points');
        pos = get(h,'Position');
        pos(1:2) = pos(1:2) + [0, -3];
        set(h,'Position', pos);
        set(h,'Units', old_units);
    end
    
    % show size and zoom factor
    if (zoom > 1)
        zformat = sprintf('* %d',round(zoom));
    else
        zformat = sprintf('/ %d',round(1/zoom));
    end
    format=[' Range: [%.3g, %.3g] \n Dims: [%d, %d] ', zformat];
    xlabel(sprintf(format, range(1), range(2), size(out_img,1), size(out_img,2)));
    h = get(gca,'Xlabel');
    set(h,'FontSize', 9);
    old_units = get(h,'Units');
    set(h,'Units','points');  
    pos = get(h,'Position');
    pos(1:2) = pos(1:2) + [0, 10];
    set(h,'Position',pos);
    set(h,'Units', old_units);
    set(h,'Visible','on');
end

return;


function [mn, mx] = range_img(img)

    ndims = length(size(img));
    if ndims == 2
        mn = min(min(img));
        mx = max(max(img));
    elseif ndims == 3
        mn = min(min(min(img)));
        mx = max(max(max(img)));
    end

return