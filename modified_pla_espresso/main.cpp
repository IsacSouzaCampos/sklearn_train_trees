#include <stdlib.h>
#include <string>
#include <iostream>
#include <map>
#include <fstream>
#include <list>
#include <vector>
#include <omp.h>

using namespace std;

void processing_order(map<int, list<int>>&);

string type = "train";
string source_name = "Benchmarks_2";
string destination_name = "Benchmarks_2_aig/ex";

int main(int argc, char** argv) {
    map<int, list<int>> elements_ordered; // ordered by inputs size
    string command{""};


    processing_order(elements_ordered);

    vector<int> final_order;
    for(auto& element : elements_ordered) {
        for(auto& lst : element.second) {
            cout << lst << endl;
            final_order.push_back(lst);
        }
    }


    omp_set_num_threads(4);
    #pragma omp parallel for
    for(int i = 0; i < final_order.size(); i++) {
        int element = final_order[i];
        if(element < 10) {
            command = "espresso "+source_name+"/ex0"
                        +to_string(element)+"."+type+".pla > "+destination_name+"0"+to_string(element)+".pla";
        }
        else {
            command = "espresso "+source_name+"/ex"
                        +to_string(element)+"."+type+".pla > "+destination_name+to_string(element)+".pla";
        }
        cout << command << endl;
        system(command.c_str());
    }

    return 0;
}

void processing_order(map<int, list<int>>& final_map) {  // map<number of inputs, ex number>

    for(int i = 0; i < 100; i++) {
        ifstream input_file;
        string line{""}, element{""};
        if(i < 10) {
            input_file.open(source_name+"/ex0"+to_string(i)+"."+type+".pla");
            element += "ex0" + to_string(i);
        } else {
            input_file.open(source_name+"/ex"+to_string(i)+"."+type+".pla");
            element += "ex" + to_string(i);
        }
        
        while(getline(input_file, line)) {
            if(line.find(".i") != string::npos) {
                line.erase(0, 2);
                // cout << "ex" << i << " / number of inputs: " << line << endl;
                final_map[stoi(line)].push_back(i);
                break;
            }
        }
    }

}
