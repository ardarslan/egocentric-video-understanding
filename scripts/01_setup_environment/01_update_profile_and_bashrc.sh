cat lines_to_add_to_the_profile.txt | cat - ~/.profile > temp && mv temp ~/.profile
cat lines_to_add_to_the_bashrc.txt | cat - ~/.bashrc > temp && mv temp ~/.bashrc
