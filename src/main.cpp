//Copyright (c) 2015 Zachary Kann
//
//Permission is hereby granted, free of charge, to any person obtaining a copy
//of this software and associated documentation files (the "Software"), to deal
//in the Software without restriction, including without limitation the rights
//to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//copies of the Software, and to permit persons to whom the Software is
//furnished to do so, subject to the following conditions:
//
//The above copyright notice and this permission notice shall be included in all
//copies or substantial portions of the Software.
//
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//SOFTWARE.

// ---
// Author: Zachary Kann

#include "z_sim_params.hpp"
#include "z_stat.hpp"
#include "z_vec.hpp"
#include "z_molecule.hpp"
#include "z_tcf.hpp"
#include "z_atom_group.hpp"
#include "z_gromacs.hpp"
#include "xdrfile_trr.h"
#include "boost/program_options.hpp"

namespace po = boost::program_options;

// Units are nm, ps.

int main (int argc, char *argv[]) {
  int st;
  SimParams params;
  int max_steps = std::numeric_limits<int>::max();
  int steps_guess = 1000;
  const int kStepsGuessIncrement = 1000;

  double r_cut;
  po::options_description desc("Options");
  desc.add_options()
    ("help,h",  "Print help messages")
    ("group,g", po::value<std::string>()->default_value("He"),
     "Group for temperature profiles")
    ("liquid,l", po::value<std::string>()->default_value("OW"),
     "Group to use for calculation of surface")
    ("index,n", po::value<std::string>()->default_value("index.ndx"),
     ".ndx file containing atomic indices for groups")
    ("gro", po::value<std::string>()->default_value("conf.gro"),
     ".gro file containing list of atoms/molecules")
    ("top", po::value<std::string>()->default_value("topol.top"),
     ".top file containing atomic/molecular properties")
    ("rcut,r", po::value<double>(&r_cut),
     "Cutoff radius for cluster membership")
    ("cont,c", "Require that solvation be continuous from t=0");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  if (vm.count("help")) {
    std::cout << desc << "\n";
    exit(EXIT_SUCCESS);
  }

  const bool cont = vm.count("cont") ? true : false;

  double r_cut_squared = r_cut*r_cut;

  std::map<std::string, std::vector<int> > groups;
  groups = ReadNdx(vm["index"].as<std::string>());

  std::vector<Molecule> molecules = GenMolecules(vm["top"].as<std::string>(),
                                                 params);
  AtomGroup all_atoms(vm["gro"].as<std::string>(), molecules);
  AtomGroup selected_group(vm["group"].as<std::string>(),
                           SelectGroup(groups, vm["group"].as<std::string>()),
                           all_atoms);
  AtomGroup liquid_group(vm["liquid"].as<std::string>(),
                         SelectGroup(groups, vm["liquid"].as<std::string>()),
                         all_atoms);

  arma::icube within = arma::zeros<arma::icube>(selected_group.size(),
                                                liquid_group.size(),
                                                steps_guess);

  rvec *x_in = NULL;
  matrix box_mat;
  arma::rowvec box = arma::zeros<arma::rowvec>(DIMS);
  std::string xtc_filename = "prod.xtc";
  XDRFILE *xtc_file;
  params.ExtractTrajMetadata(strdup(xtc_filename.c_str()), (&x_in), box);
  xtc_file = xdrfile_open(strdup(xtc_filename.c_str()), "r");
  params.set_max_time(vm["max_time"].as<double>());

  arma::rowvec dx;
  float time,prec;
  int step;
  for (step=0; step<max_steps; step++) {
    if(read_xtc(xtc_file, params.num_atoms(), &st, &time, box_mat, x_in, &prec))
      break;
    if (step == steps_guess) {
      steps_guess += kStepsGuessIncrement;
      within.resize(selected_group.size(), liquid_group.size(),
                    steps_guess);
    }
    params.set_box(box_mat);
    int i = 0;
    for (std::vector<int>::iterator i_atom = selected_group.begin();
         i_atom != selected_group.end(); i_atom++, i++) {
      selected_group.set_position(i, x_in[*i_atom]);
    }
    i = 0;
    for (std::vector<int>::iterator i_atom = liquid_group.begin();
         i_atom != liquid_group.end(); ++i_atom, ++i) {
      liquid_group.set_position(i, x_in[*i_atom]);
    }
    for (int i_atom = 0; i_atom < selected_group.size(); ++i_atom) {
      for (int i_liq = 0; i_liq < liquid_group.size(); ++i_liq) {
        if (selected_group.index_to_molecule(i_atom) ==
            liquid_group.index_to_molecule(i_atom)) {
          continue;
        }
        FindDxNoShift(dx, selected_group.position(i_atom),
                      liquid_group.position(i_liq), box);
        double r2 = arma::dot(dx,dx);
        if (r2<r_cut_squared)
          within(i_atom,i_liq,step)++;
      }
    }
  }

  TCF<int> tcf(1000);
  double num_corr = (step - tcf.length())/tcf.interval() + 1;
  arma::irowvec counter = arma::zeros<arma::irowvec>(tcf.length());

  for (int i_atom = 0; i_atom < selected_group.size(); ++i_atom) {
    for (int i_num = 0; i_num < num_corr; i_num++) {
      for (int i_liq = 0; i_liq < selected_group.size(); ++i_liq) {
        bool cont_check = true;
        for (int i_corr = 0; i_corr < tcf.length(); i_corr++) {
          int i_step = i_num + i_corr;
          if (within(i_atom,i_liq,i_step)) {
            if (!cont || cont_check)
              counter(i_corr)++;
            else
              cont_check = false;
          }
          tcf.CorrelateOneDirection(counter);
        }
      }
    }
  }

  for (int i_atom = 0; i_atom < selected_group.size(); ++i_atom) {
    double slope = ExponentialFitSlope(params.dt(), counter.row(i_atom), 200,
                                       tcf.length());
    std::cout << -slope << std::endl;
  }
}
 // main
