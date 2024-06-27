#include <iostream>
#include <fstream>
#include <string>
#include <limits>
#include <filesystem>

namespace fs = std::filesystem;
int inputnum;
void double2integer(const std::string& filename) {
	std::cout << "the file is: "<<filename<< std::endl; 
	std::cin >> inputnum; 
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Unable to open input file " << filename << std::endl;
        return;
    }

    std::string newfilename = filename+ "new";
    std::ofstream outfile(newfilename);
    if (!outfile.is_open()) {
        std::cerr << "Unable to open output file " << newfilename << std::endl;
        file.close();
        return;
    }
    std::string line;
	int count=0; 
	int count_round = 0;
    while (std::getline(file, line)) {
        try {
			if(count==0){
				 outfile<<"wire signed [width-1:0] weight" << count_round <<" [0:input_number-1]= '{";
				 count_round++; 
			}
            double value = std::stod(line);
            int num_int = static_cast<int>(1024 * value);
            if (num_int >= 2048) num_int = 2048;
			if (num_int <= -2048) num_int = -2048;
            outfile << num_int;
			if(count==(inputnum-1)) outfile << "};" << std::endl; 
			else outfile << ","; 
			count = (count+1)%inputnum; 
        } catch (const std::invalid_argument& e) {
            std::cerr << "Invalid number in file: " << line << std::endl;
            continue;
        } catch (const std::out_of_range& e) {
            std::cerr << "Number out of range in file: " << line << std::endl;
            continue;
        }
    }
	outfile<<std::endl;
    outfile.close();
    file.close();
}

int main() {
    std::string name;
    std::cout << "Input the beginning string of the filename in this directory: ";
    std::cin >> name;
    std::string directory_path = ".";  // current directory

    for (const auto& entry : fs::directory_iterator(directory_path)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().string();
            if (filename.find(name) == 0) {  // Match files starting with 'name'
                double2integer(entry.path().string());
                std::cout << filename << " finished transformation" << std::endl;
            }
        }
    }

    return 0;
}

