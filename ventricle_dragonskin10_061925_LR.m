jclear all; close all; clc;

% Setup parameters

pr = [-0.0002, 0.0244  0.3396, 0.4661];
pz = [0.0003, 0.0035, 0.0941, -0.2788];

%13 Dec video
pr = [-0.0004, 0.0259,  0.1868, 0.8807];
pz = [0.0001, 0.0066, 0.0159, -0.0666];

uRange = [0 13*2];

% Axis info
% Name axis according to appropriate naming convension
printBedAxis = "X";
ssAxis = "Y";
rotaryAxis = "Z";

r_min = 1; % The smallest radii the robots can work on

% Distance between printed layers
lineDiameter = .8;
nitinolDiam = 1;
step = nitinolDiam * 0.5; % 0.5 for dragonskin10, 0.46 for ecoflex

% Print info
printSpeed = 10; %20; % mm/min
syringeDiameter = 12.07; % 5mL syringe, mL


% Ellipsoid parameters
% Index 1 is the outermost ellipsoid
a_array = [13, 13-lineDiameter*.75, 7.5];
c_array = [30.5, 30.5-lineDiameter*.75, 22.5];
extrusionAxisEllipsoid = {"B", "A", "A"};

% Spiral parameters
a_spiral = 13-(lineDiameter*.75*2);
c_spiral = 30.5;
z_spiral_start = 5;
spiral_revolutions = 1.0; %originally 0.67, then 1.5, now 1.25. 1.0 since trying add circular ones

number_of_splines = 10; %originally 12
extrusionAxisSpirals = "C";

% Roof parameters
n_layers_roof = 2;
extrusionAxisRoof = {"B", "C"};

%% Automated script

z_top = c_array(1);

figure(1);
hold on;
axis equal
view(3)

syms v

% ========= Ellipsoids =========

for k = 1:length(a_array)-1
    a = a_array(k);
    c = c_array(k);
    eqn = (r_min == a*sin(v));
    s = double(vpasolve(eqn, v, [0, pi/2]));
    s_prev = s;
    r0 = a * sin(s);
    z0 = z_top - c * cos(s);
    pts_add = [r0, z0];
    zz = z0;

    while zz + step < z_top
        s = fmincon(@(x) findNextPt(x, pts_add(end, :), [a, c, z_top], step), ...
            s_prev+eps, [], [], [], [], s_prev, pi/2);
        s_prev = s;
        rr = a * sin(s);
        zz = z_top - c * cos(s);
        pts_add = [pts_add;
            [rr, zz]
            ];
    end
    ellipsoid_pts{k} = pts_add;
    plot3(pts_add(:,1), zeros(size(pts_add(:,1))), pts_add(:,2), 'o-')
    filename = "ellipsoid_" + string(k) + ".txt";
    writeGCodeCircles(pts_add, printBedAxis, ssAxis, rotaryAxis, pr, pz, filename, uRange, lineDiameter, printSpeed, extrusionAxisEllipsoid{k}, syringeDiameter)


    %% sample
    
    pts_add_c_stiff_0 = [pts_add(15,:); pts_add(30,:); pts_add(45,:); pts_add(60,:)];
    
    [~, idx1] = min(abs(pts_add(:,2) - z_spiral_start));
    [~, idx2] = min(abs(pts_add(:,2) - pts_add(60,2))); 

    
    val2_k = 60/length(pts_add); %0.88

    proximity_k_circle = linspace(0.7,0.9, length(pts_add_c_stiff_0)); %values may need to be adjusted for silicone
    proximity_k_circle = proximity_k_circle';

    pts_add_c_stiff = [pts_add_c_stiff_0(:,1).*proximity_k_circle pts_add_c_stiff_0(:,2)];
end

% ========= Roof =========

for k = 1:n_layers_roof

    if k == 1
        n = ceil([a_array(1) - (a_array(1)-3)] / (step/1.5));
        rr = linspace(a_array(1)+(step/2), a_array(1)-3, n);
        zz = ones(size(rr))*(z_top + (k-1)*(step/1.5));

    else
        n = ceil((a_array(1) -1) / (step/1.5));
        rr = linspace(a_array(1)+(step/2), 1, n);
        zz = ones(size(rr))*(z_top-0.5 + (k-1)*(step/1.5));

    end

    roof_pts{k} = [rr', zz'];
    plot3(roof_pts{k}(:,1), zeros(size(roof_pts{k}(:,1))), roof_pts{k}(:,2), 'o-')
    filename = "roof_" + string(k) + ".txt";
    writeGCodeCircles(roof_pts{k}, printBedAxis, ssAxis, rotaryAxis, pr, pz, filename, uRange, lineDiameter, printSpeed, extrusionAxisRoof{k}, syringeDiameter)
end

% ========= Spirals =========

angle_increment = 360 / number_of_splines;
a = a_spiral;
c = c_spiral;

n = 100;
theta = linspace(0, spiral_revolutions * 2 * pi, n);
phi = linspace(0, pi/2, n);

proximity_k = linspace(0.68,0.92, n); %constant 0.85 is too close at the apex and too far from the base.
    
x = a .* cos(theta) .* sin(phi.*proximity_k);
y = a .* sin(theta) .* sin(phi.*proximity_k);
z = z_top - c .* cos(phi);
inds = z > z_spiral_start;
pts_stiffeners{1} = [x(inds)', y(inds)', z(inds)'];
plot3(pts_stiffeners{1}(:,1),pts_stiffeners{1}(:,2),pts_stiffeners{1}(:,3),'k');
filename = "stiffener_01.txt";
writeGCodeSpiral(pts_stiffeners{1}, printBedAxis, ssAxis, rotaryAxis, pr, pz, filename, uRange, lineDiameter, printSpeed, extrusionAxisSpirals, syringeDiameter)

for k = 2:number_of_splines
    R = rotz(angle_increment);
    pts_stiffeners{k} = (R * pts_stiffeners{k-1}')';
    plot3(pts_stiffeners{k}(:,1),pts_stiffeners{k}(:,2),pts_stiffeners{k}(:,3),'o');
    plot3(pts_stiffeners{k}(1,1),pts_stiffeners{k}(1,2),pts_stiffeners{k}(1,3),'kx');
    filename = "stiffener_" + sprintf('%02d',k) + ".txt";
    writeGCodeSpiral(pts_stiffeners{k}, printBedAxis, ssAxis, rotaryAxis, pr, pz, filename, uRange, lineDiameter, printSpeed, extrusionAxisSpirals, syringeDiameter)
end

plot3(pts_add_c_stiff(:,1), zeros(size(pts_add_c_stiff(:,1))), pts_add_c_stiff(:,2),'k*','MarkerSize',10)
filename = "circles_stiffer" + ".txt";
writeGCodeStiffCirc(pts_add_c_stiff, printBedAxis, ssAxis, rotaryAxis, pr, pz, filename, uRange, lineDiameter, printSpeed, extrusionAxisSpirals, syringeDiameter)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Combine Stiffeners into single file
% Specify the directory containing the files (use pwd for current folder)
folder_path = pwd;
% Specify the pattern for your text files
file_pattern = fullfile(folder_path, 'stiffener_*.txt'); % Matches 'stiffener_2.txt', 'stiffener_3.txt', etc.

% Get a list of all matching files
text_files = dir(file_pattern);

% Sort the files by name (optional, ensures sequential processing)
text_files = sort({text_files.name});

% Find the index of '04.txt'
startIndex = find(strcmp(text_files, 'stiffener_05.txt'));

% Reorder the list so that it starts with '04.txt'
text_files = [text_files(startIndex:end), text_files(1:startIndex-1)];

% Specify the name of the output file
output_filename = fullfile(folder_path,'stiffeners_comb.txt');
% Open the output file for writing
fid_out = fopen(output_filename, 'w');

if fid_out == -1
    error('Could not open output file for writing.');

end

% Loop through all text files
for i = 1:length(text_files)
    % Get the full path of the current file
    current_file = fullfile(folder_path, text_files{i});
    % Read the content of the current file as a table, remove first 2 lines
    data = readtable(current_file, 'ReadVariableNames', false);

    % If Z switches form 2nd to 3rd quadrant, add 360
    if i > 8 % was 6 for 1.5 rev; needs to be 7 for 1.25; 8 for 1.0?
        Z_col_idx = find(startsWith(string(data{1, :}), 'Z'), 1); % Find the column index for 'Z'
        if ~isempty(Z_col_idx)
            Z_column = data{:, Z_col_idx}; % Extract the 'Z' column
            for k = 1:length(Z_column)
                % Extract the numeric part after 'Z'
                value = str2double(extractAfter(Z_column{k}, 'Z'));
                % Add 360 to the numeric part
                new_value = value + 360;
                % Create the new string with 'Z' and the updated value
                Z_column{k} = sprintf('Z%.4f', new_value);
            end
            % Update the data table with the modified Z_column
            data{:, Z_col_idx} = Z_column;
        end
    end

    % If the file index is even, reverse the rows except the 'extrusionAxisSpirals' column
    if mod(i, 2) == 1
        % Identify the column starting with 'extrusionAxisSpirals'
        C_col_idx = find(startsWith(string(data{1, :}), extrusionAxisSpirals), 1);
        if ~isempty(C_col_idx)
            % Extract the 'extrusionAxisSpirals' column
            C_column = data{:, C_col_idx};
            % Reverse the rest of the table
            data = flip(data, 1);
            % Restore the original 'extrusionAxisSpirals' column
            data{:, C_col_idx} = C_column;
        else
            data = flip(data, 1);
        end
    end

    if i==1
        fprintf(fid_out,'%s\n','G92 Z1');
        fprintf(fid_out,'%s\n','G90 G1 X-29.5 Y15.25 Z320 F350');
        fprintf(fid_out,'%s\n','G91 G1 C7 F400'); %change if changing axis; was 14 now 7
    end

    fprintf(fid_out, 'G92 %s0\n', extrusionAxisSpirals);
    % fprintf(fid_out, 'G91 %s0\n', 'G1', extrusionAxisSpirals, '14');
    for row = 1:height(data)
        fprintf(fid_out,'%s\n',strjoin(string(data{row,:}),' '));
    end
end

% Close the output file
fclose(fid_out);
disp(['Files combined into: ' output_filename]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%

output_filename = fullfile(folder_path,'fullventricle.txt');
files = {'ellipsoid_1.txt', 'roof_1.txt', 'ellipsoid_2.txt', 'stiffeners_comb.txt', 'circles_stiffer', 'roof_2.txt'};
fid_out = fopen(output_filename, 'w');
for i=1:length(files)
    % Get the full path of the current file
    current_file = fullfile(folder_path, files{i});
    % Read the content of the current file as a table
    data = readtable(current_file,'NumHeaderLines',0,'ReadVariableNames',false);
    if i==1
        %fprintf(fid_out,'%s\n','\\ Ellipsoid 1');
    end

    for row = 1:height(data)
        fprintf(fid_out,'%s\n',strjoin(string(data{row,:}),' '));
    end

    if i==2
        fprintf(fid_out,'%s\n',' ');
        fprintf(fid_out,'%s\n',strjoin("G91 G1 "+extrusionAxisEllipsoid{1}+"-1.5 F250"));
        fprintf(fid_out,'%s\n',strjoin("G91 G1 X-75 F250"));
        fprintf(fid_out,'%s\n',strjoin("G91 G1 Z360 F600"));
        fprintf(fid_out,'%s\n',strjoin("G91 G1 "+extrusionAxisEllipsoid{2}+"280 F250"));
        fprintf(fid_out,'%s\n',strjoin("G91 G1 X75 F250"));
        fprintf(fid_out,'%s\n',' ');

    elseif i==3
        fprintf(fid_out,'%s\n',' ');
        fprintf(fid_out,'%s\n',strjoin("G91 G1 "+extrusionAxisEllipsoid{2}+"-10.5 F250"));
        fprintf(fid_out,'%s\n',strjoin("G91 G1 X-75 F250"));
        fprintf(fid_out,'%s\n',strjoin("G91 G1 Z360 F600"));
        fprintf(fid_out,'%s\n',strjoin("G91 G1 "+extrusionAxisSpirals+"280 F250"));
        fprintf(fid_out,'%s\n',strjoin("G91 G1 X75 F250"));
        fprintf(fid_out,'%s\n',' ');
    elseif i==4
        fprintf(fid_out,'%s\n',' ');
        % fprintf(fid_out,'%s\n',strjoin("G91 G1 "+extrusionAxisSpirals+"-10.5 F250"));
        % fprintf(fid_out,'%s\n',strjoin("G91 G1 X-75 F250"));
        % fprintf(fid_out,'%s\n',strjoin("G91 G1 A-450 F1000"));
        % fprintf(fid_out,'%s\n',strjoin("G91 G1 Z360 F600"));
        % fprintf(fid_out,'%s\n',strjoin("G91 G1 "+extrusionAxisRoof{2}+"280 F250"));
        % fprintf(fid_out,'%s\n',strjoin("G91 G1 X75 F250"));
        % fprintf(fid_out,'%s\n',' ');
    else
        fprintf(fid_out,'%s\n',' ');
        fprintf(fid_out,'%s\n',' ');
    % else
    %     fprintf(fid_out,'%s\n',' ');
    %     fprintf(fid_out,'%s\n',strjoin("G91 G1 "+extrusionAxisEllipsoid{3}+"-1.5 F250"));
    %     fprintf(fid_out,'%s\n','G90 G1 X-75 F600');
    %     fprintf(fid_out,'%s\n','G90 G1 Y1 F400');
    end

end
fclose(fid_out);

function c = findNextPt(x, prevPt, params, desiredDist)
rNext = params(1) * sin(x);
zNext = params(3) - params(2) * cos(x);
dist = sqrt((rNext-prevPt(1))^2 + (zNext-prevPt(2))^2);
c = (dist - desiredDist)^2;
end


function writeGCodeCircles(pts, printBedAxis, ssAxis, rotaryAxis, pr, pz, filename, uRange, lineDiameter, printSpeed, extrusionAxis, syringeDiameter)
syms u
fid = fopen(filename, "w");
fprintf(fid, "G92 " + extrusionAxis + "0\n");
extrusionPos = 0;

x_prior = 0;
y_prior = 0;

for i = 1:size(pts,1)
    eqn = pr(1)*u^3 + pr(2)*u^2 + pr(3)*u + pr(4) == pts(i,1);
    s = double(vpasolve(eqn, u, uRange));
    s = round(s, 3);

    z_rel = pz(1)*s^3 + pz(2)*s^2 + pz(3)*s + pz(4);
    z_rel = round(z_rel, 3);
    z_abs = pts(i,2);
    z_abs = -round(z_abs, 3);

    a = abs(z_rel);
    b = -1*sign(z_abs)*z_abs;

    % Calculate angular velocity F (in deg/min) for constant linear velocity
    angularVelocity = (10800 * printSpeed / (pi * s));


    extrusionPos = 2*pi*pts(i,1)*(lineDiameter/syringeDiameter)^2;
    
 
    if extrusionAxis == "A" || extrusionAxis == "D"
        silicone_overextrusion = 1.3; %% 1.3 with dragonskin, 1.43 with ecoflex
        extrusionPos = extrusionPos * 14 * silicone_overextrusion;
        extrusion_vel = num2str(min(2200,angularVelocity));
    elseif  extrusionAxis == "C"
        extrusion_vel = "2000";
    else
        extrusion_vel = "8000";
    end

    abs_x = a+b-2.5;
    abs_y = s;

    if i == 1

        string_add_lin = "G90 G1" + ...
            " " + printBedAxis + string(abs_x - x_prior) + ...
            " " + ssAxis + string(abs_y - y_prior) + ...
            " F" + "400" + ...
            "\n";
        fprintf(fid, string_add_lin);

    else
        string_add_lin = "G91 G1" + ...
            " " + printBedAxis + string(abs_x - x_prior) + ...
            " " + ssAxis + string(abs_y - y_prior) + ...
            " " + rotaryAxis + "360" + ...
            " " + extrusionAxis + extrusionPos*(1+2/pts(i,1)^1.5) + ...
            " F" + extrusion_vel + ...
            "\n";
        fprintf(fid, string_add_lin);
    end

    x_prior = abs_x;
    y_prior = abs_y;
end
fclose(fid);
end

function writeGCodeSpiral(pts, printBedAxis, ssAxis, rotaryAxis, pr, pz, filename, uRange, lineDiameter, printSpeed, extrusionAxis, syringeDiameter)
syms u
fid = fopen(filename, "w");
fprintf(fid,"G90\n");
fprintf(fid, "G92 " + extrusionAxis + "0\n");
r = sqrt(pts(:,1).^2 + pts(:,2).^2);
th = atan2(pts(:,2), pts(:,1));
th = unwrap(th);
th = round(th,3);
extrusionPos_i = 0;

for i = 2:size(pts,1)
    eqn = pr(1)*u^3 + pr(2)*u^2 + pr(3)*u + pr(4) == r(i);
    s = double(vpasolve(eqn, u, uRange));
    s = round(s, 3);

    z_rel = pz(1)*s^3 + pz(2)*s^2 + pz(3)*s + pz(4);
    z_rel = round(z_rel, 3);
    z_abs = pts(i,3);
    z_abs = -round(z_abs, 3);

    dist = norm(pts(i,:)-pts(i-1,:));

    % extrusion_rel = abs(dist * (lineDiameter / syringeDiameter)^2 )* 2
    extrusion_rel = (dist^0.33) * (lineDiameter^2 / syringeDiameter^2);
    silicone_overextrusion = 3.8; %was 3.15, then 4.05 (silpoxy), now 3.3 (dragonskin)

    if extrusionAxis == "A" || extrusionAxis == "D" || extrusionAxis == "C"
        extrusion_rel = extrusion_rel * 14 * silicone_overextrusion;
    end

    extrusionPos = extrusionPos_i + extrusion_rel;

    dth = abs(th(i)-th(i-1))*180/pi;
    rotationSpeed = dth*printSpeed/dist;

    string_add = "G90 G1" + ...-
        " " + rotaryAxis + th(i)*180/pi + ...
        " " + printBedAxis + string(z_rel + z_abs-2.5) + ...
        " " + ssAxis + string(s) + ...
        " " + extrusionAxis + string(extrusionPos) + ...
        " F" + "725" + ... (was 666.666 then 725 (silpoxy) now 700 but motor sticking so 600 now 650 (dragonskin))
        "\n";

    extrusionPos_i = extrusionPos;
    fprintf(fid, string_add);

    % fprintf("Step %d: dist = %.3f, extrusion_rel = %.3f, extrusionPos = %.3f\n", i, dist, extrusion_rel, extrusionPos);

end
fclose(fid);
end

function writeGCodeStiffCirc(pts, printBedAxis, ssAxis, rotaryAxis, pr, pz, filename, uRange, lineDiameter, printSpeed, extrusionAxis, syringeDiameter)
syms u
fid = fopen(filename, "w");
fprintf(fid, "G92 " + extrusionAxis + "0\n");
extrusionPos = 0;

x_prior = 0;
y_prior = 0;

for i = 1:size(pts,1)
    eqn = pr(1)*u^3 + pr(2)*u^2 + pr(3)*u + pr(4) == pts(i,1);
    s = double(vpasolve(eqn, u, uRange));
    s = round(s, 3);

    z_rel = pz(1)*s^3 + pz(2)*s^2 + pz(3)*s + pz(4);
    z_rel = round(z_rel, 3);
    z_abs = pts(i,2);
    z_abs = -round(z_abs, 3);

    a = abs(z_rel);
    b = -1*sign(z_abs)*z_abs;

    % Calculate angular velocity F (in deg/min) for constant linear velocity
    angularVelocity = (10800 * printSpeed / (pi * s));


    extrusionPos = 3.5*pi*pts(i,1)*(lineDiameter/syringeDiameter)^2;
    
 
    if extrusionAxis == "A" || extrusionAxis == "C" || extrusionAxis == "D"
        silicone_overextrusion = 2; %% 1.3 with dragonskin, 1.43 with ecoflex
        extrusionPos = extrusionPos * 14 * silicone_overextrusion;
        extrusion_vel = "1000";
    elseif  extrusionAxis == "C"
        extrusion_vel = "1000";
    else
        extrusion_vel = "8000";
    end

    abs_x = a+b-2.5;
    abs_y = s;

    if i == 1

        string_add_lin1 = "G90 G1" + ...
            " " + printBedAxis + string(abs_x - x_prior) + ...
            " " + ssAxis + string(abs_y - y_prior) + ...
            " F" + extrusion_vel + ...
            "\n";
                string_add_lin2 = "G91 G1" + ...
            " " + rotaryAxis + "365" + ...
            " " + extrusionAxis + extrusionPos*(1+2/pts(i,1)^1.5) + ...
            " F" + extrusion_vel + ...
            "\n";
        fprintf(fid, string_add_lin1);
        fprintf(fid, string_add_lin2);

    else
        string_add_lin1 = "G91 G1" + ...
            " " + printBedAxis + string(abs_x - x_prior) + ...
            " " + ssAxis + string(abs_y - y_prior) + ...
            " F" + extrusion_vel + ...
            "\n";
                string_add_lin2 = "G91 G1" + ...
            " " + rotaryAxis + "365" + ...
            " " + extrusionAxis + extrusionPos*(1+2/pts(i,1)^1.5) + ...
            " F" + extrusion_vel + ...
            "\n";
        fprintf(fid, string_add_lin1);
        fprintf(fid, string_add_lin2);
    end

    x_prior = abs_x;
    y_prior = abs_y;
end
fclose(fid);
end

%% I used 1.4g of Thivex for 4+4 (A+B) Ecoflex --> Changed to usual 0.4 and worked better; added no thinner (can try add)

%% REMEMBER TO HAVE ROOF PRINTED WITH D WITHOUT PAUSE OR ANYTHING

%% REMEMBER TO REMOVE FIRST ELLIPSOID (TISSUE PRINTED IN B)