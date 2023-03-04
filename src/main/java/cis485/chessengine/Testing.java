package cis485.chessengine;

import cis485.chessengine.Engine.Engine;
import com.github.bhlangonijr.chesslib.Board;
import com.github.bhlangonijr.chesslib.Piece;
import com.github.bhlangonijr.chesslib.Side;
import com.github.bhlangonijr.chesslib.Square;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.HashMap;

public class Testing {
    public static void main(String[] args) {
        MultiLayerNetwork model = null;
        try {
            model = MultiLayerNetwork.load(new File("C:\\Users\\drewm\\Desktop\\EngineModels\\SL_MODEL_V1.mdl"), false);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        Engine engine = new Engine(model);
        engine.setSecondsPerMove(10f);
        engine.setSide(Side.WHITE);

        Board board = new Board();

        BufferedImage boardImage;
        HashMap<String, BufferedImage> pieceImages = new HashMap<>();
        try {
            boardImage = ImageIO.read(new File("C:\\Users\\drewm\\Desktop\\ChessEngineUI\\Images\\board.png"));
            File dir = new File("C:\\Users\\drewm\\Desktop\\ChessEngineUI\\Images\\Pieces");
            File[] files = dir.listFiles();
            for (File file : files) {
                pieceImages.put(file.getName().substring(0, file.getName().length() - 4), ImageIO.read(file));
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        JFrame frame = new JFrame("Chess Engine");
        frame.setSize(512, 600);
        frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);

        JPanel boardPanel = new JPanel() {
            @Override
            protected void paintComponent(Graphics g) {
                super.paintComponent(g);
                g.drawImage(boardImage, 0, 0, this);
                for (int x = 0; x < 8; x++) {
                    for (int y = 0; y < 8; y++) {
                        int px = x * 60;
                        int py = (Math.abs(y - 8) - 1) * 60;
                        Piece piece = board.getPiece(Square.squareAt(x + y * 8));
                        if (piece != Piece.NONE) {
                            g.drawImage(pieceImages.get(piece.name()), px, py, this);
                        }
                    }
                }
            }
        };

        JPanel buttonsPanel = new JPanel();
        JButton runEngineButton = new JButton("Run Engine");
        runEngineButton.addActionListener(e -> {
            System.out.println(board);
            engine.setSide(board.getSideToMove());
            board.doMove(engine.run(board.getFen()));
            engine.printEvaluations();
            frame.repaint();
        });

        TextField moveTextField = new TextField();
        moveTextField.setColumns(4);

        JButton doMoveButton = new JButton("Do");
        doMoveButton.addActionListener(e -> {
            System.out.println(board);
            String move = moveTextField.getText();
            moveTextField.setText("");
            board.doMove(move);
            frame.repaint();
        });

        buttonsPanel.add(moveTextField);
        buttonsPanel.add(doMoveButton);
        buttonsPanel.add(runEngineButton);


        frame.setLayout(new BorderLayout());
        frame.add(boardPanel, BorderLayout.CENTER);
        frame.add(buttonsPanel, BorderLayout.SOUTH);
        frame.setVisible(true);
    }
}
